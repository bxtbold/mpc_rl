import torch
import torch.nn as nn

from mpc_rl.network.mlp import MLP


class Dynamics(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = ... # non-learanble actual dynamic goes here

    def forward(self, state, action):
        # residual dynamics: x_{t+1} = x_t + f(x_t, action)
        delta_state = self.net(
            torch.cat([state, action], dim=1)
        )
        return state + delta_state


class LearnableDynamics(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, hidden_dim=16):
        super().__init__()
        self.net = MLP(input_dim, output_dim, num_layers, hidden_dim)

    def forward(self, state, action):
        # residual dynamics: x_{t+1} = x_t + f(x_t, action)
        delta_state = self.net(
            torch.cat([state, action], dim=1)
        )
        return state + delta_state

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)
