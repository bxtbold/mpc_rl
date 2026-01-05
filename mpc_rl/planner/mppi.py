from collections.abc import Callable
from typing import Optional

import numpy as np
import torch
from torch import Tensor


class MPPI(torch.nn.Module):
    def __init__(
        self,
        horizon: int,
        nx: int,
        nu: int,
        dynamics: Callable,
        running_cost: Callable,
        terminal_cost: Callable | None=None,
        num_samples: int=1000,
        lambda_: float=1.0,
        noise_sigma: float=1.0,
        u_min: Tensor | None=None,
        u_max: Tensor | None=None,
        device: str="cuda",
    ):
        """
        Initialize the MPPI planner.

        Args:
            horizon: planning horizon
            nx: state dimension
            nu: action dimension
            dynamics: dynamics function
            running_cost: running cost function
            terminal_cost: terminal cost function
            num_samples: number of samples
            lambda_: lambda parameter
            noise_sigma: noise standard deviation
            u_min: minimum action
            u_max: maximum action
            device: device to run on
        """
        super().__init__()
        self.device = device
        self.dynamics = dynamics
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost

        self.nx = nx
        self.nu = nu
        self.H = horizon
        self.K = num_samples
        self.lambda_ = lambda_
        self.sigma = noise_sigma

        self.u_min = u_min.to(device)
        self.u_max = u_max.to(device)

        # nominal control sequence (H, nu)
        self.U = torch.zeros(self.H, self.nu, device=device)

        # precompute inverse covariance (diagonal)
        self.inv_sigma2 = 1.0 / (self.sigma ** 2)
        print("MPPI is initialized.")

    def get_action(self, x: Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return self(x).cpu().detach().numpy()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.device)

        # sample noise
        eps = torch.randn(self.K, self.H, self.nu, device=self.device) * self.sigma

        # apply perturbations
        U_nom = self.U.unsqueeze(0)  # (1, H, nu)
        U_samples = U_nom + eps
        U_samples = torch.clamp(U_samples, self.u_min, self.u_max)

        # compute costs
        costs = self._rollout_cost(x, U_samples)  # (K,)

        # importance weights
        w = torch.exp(-(costs - costs.min()) / self.lambda_)
        w = w / (w.sum() + 1e-8)

        # MPPI update: weighted perturbations
        delta_u = torch.einsum("k,khn->hn", w, eps)
        self.U = torch.clamp(self.U + delta_u, self.u_min, self.u_max)

        # receding horizon
        u0 = self.U[0].clone()
        self.U[:-1] = self.U[1:].clone()
        self.U[-1].zero_()

        return u0

    def _rollout_cost(self, x0: Tensor, U_samples: Tensor) -> Tensor:
        K, H, _ = U_samples.shape
        x = x0.unsqueeze(0).expand(K, -1).clone()

        total_cost = torch.zeros(K, device=self.device)

        for t in range(H):
            u = U_samples[:, t, :]
            total_cost += self.running_cost(x, u)
            x = self.dynamics(x, u)

        if self.terminal_cost is not None:
            total_cost += self.terminal_cost(x)

        return total_cost

    def reset(self):
        self.U.zero_()
