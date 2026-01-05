import time

import tqdm

from mpc_rl.planner.mppi import MPPI
from mpc_rl.tasks.pendulum import dynamics, make_env, running_cost


def main(args):
    env = make_env(render_mode="human")

    planner = MPPI(
        horizon=args.horizon,
        nx=env.observation_space.shape[0],
        nu=env.action_space.shape[0],
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=None,
        num_samples=args.num_samples,
        lambda_=args.lambda_,
        noise_sigma=args.noise_sigma,
        u_min=env.action_space.low,
        u_max=env.action_space.high,
        device=args.device,
    )
    print(planner)

    env.reset()
    for _ in tqdm.tqdm(range(args.num_steps), desc="Simulation steps"):
        state = env.unwrapped.state.copy()  # this returns [th, thdot]
        # on the other hand, env.get_obs() returns [sin(th), cos(th), thdot]

        action = planner.get_action(state)
        env.step(action)
        env.render()
        time.sleep(0.01)

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--lambda_", type=float, default=1.0)
    parser.add_argument("--noise_sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=200)

    args = parser.parse_args()

    main(args)
