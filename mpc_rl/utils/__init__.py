
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def populate_replay_buffer(env, buffer, num_episode, episode_length, seed=1):
    for _ in range(num_episode):
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        for _ in range(episode_length):
            action = env.action_space.sample()
            next_obs, reward, truncated, terminated, _ = env.step(action)
            done = truncated or terminated
            transition = (obs, action, reward, next_obs, done)
            buffer.add(transition)
    return buffer


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable for reproducibility


def to_torch(arr, device="cuda"):
    return torch.from_numpy(arr).float().to(device)


def to_torch_batch(batch, device="cuda"):
    obs, actions, reward, next_obs, done = batch

    obs = torch.from_numpy(obs).float().to(device)
    actions = torch.from_numpy(actions).float().to(device)
    reward = torch.from_numpy(reward).float().to(device)
    next_obs = torch.from_numpy(next_obs).float().to(device)
    done = torch.from_numpy(done).float().to(device)

    return obs, actions, reward, next_obs, done


def weight_init(m, type="xavier", bias=1e-4):
    if isinstance(m, torch.nn.Linear) and type == "kaiming":
        # for mlp, cnn, etc with relu activations
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, bias)
    elif isinstance(m, torch.nn.Linear) and type == "xavier":
        # for other networks with tanh/sigmoid activations
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, bias)


def log_losses(losses: dict[str, list[float]]):
    plt.figure()
    plt.plot(losses["train"], label="train")

    eval_x = [i * 10 for i in range(len(losses["eval"]))]
    plt.plot(eval_x, losses["eval"], label="eval")

    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi
