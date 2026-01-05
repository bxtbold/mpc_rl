import gymnasium as gym
import torch

from mpc_rl.utils import angle_normalize

ENV_NAME = "Pendulum-v1"


def make_env(render_mode: str = "human"):
    return gym.make(ENV_NAME, render_mode=render_mode)


def dynamics(state, action, info={}):
    """
    Dynamics for the pendulum task.

    The dynamics are computed as:
        x_{t+1} = x_t + x_dot * dt
        x_dot_{t+1} = x_dot_t + x_ddot * dt
        x_ddot = -3 * g / (2 * l) * sin(x + pi) + 3.0 / (m * l**2) * u

    Args:
        state: state
        action: action
        info: additional information
            {
                g: gravity
                m: mass
                l: length
                dt: time step
            }
    Returns:
        state: next state
    """
    # dynamics from gymnasium
    x = state[:, 0].view(-1, 1)
    x_dot = state[:, 1].view(-1, 1)

    g = info.get("g", 10)
    m = info.get("m", 1)
    l = info.get("l", 1)
    dt = info.get("dt", 0.05)
    u = action[:, 0].view(-1, 1)

    x_ddot = -3 * g / (2 * l) * torch.sin(x + torch.pi) + 3.0 / (m * l**2) * u
    x_dot = x_dot + x_ddot * dt
    x = x + x_dot * dt

    return torch.cat((x, x_dot), dim=1)


def running_cost(state, action, info={}):
    """
    Running cost for the pendulum task.

    The cost is computed as:
        cost = w_x * angle_normalize(x) ** 2 +
               w_x_dot * x_dot**2 +
               w_action * action**2

    Args:
        state: state
        action: action
        info: additional information
            {
                w_x: weight for the angle
                w_x_dot: weight for the angular velocity
                w_action: weight for the action
            }
    Returns:
        cost: running cost
    """
    x = state[:, 0]
    x_dot = state[:, 1]
    w_x = info.get("w_x", 1.0)
    w_x_dot = info.get("w_x_dot", 0.1)
    w_action = info.get("w_action", 0.1)
    x = x.view(-1)
    x_dot = x_dot.view(-1)
    action = action.view(-1)
    cost = w_x * angle_normalize(x)**2 + w_x_dot * x_dot**2 + w_action * action**2
    return cost
