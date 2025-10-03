import torch
import numpy as np
import time
import argparse
from train import Net   # Reuse the architecture + eta/balance option arrays
from env import ContinuousWorldParallelEnv


def build_env(render: bool, max_cycles: int | None):
    env = ContinuousWorldParallelEnv(
        n_agents=2,
        max_cycles=1000,            # will override below if needed
        world_size=(10, 10),
        render_mode="human" if render else None,
        dc=0.5,
        ds=2.0,
        evader_speed=400,
        scale=100,
        pursuer_speed=300
    )
    if max_cycles is not None:
        env.max_cycles = max_cycles
    return env


def test_model(
    model_path: str,
    num_episodes: int = 5,
    render: bool = False,
    max_cycles_override: int | None = None,
    step_sleep: float = 0.0,
    log_interval: int = 200,
    batch_action: bool = True,
    quiet: bool = False,
):
    """
    Test a trained DQN-style model controlling 2 agents.

    Improvements over original:
    - Optional batching of the two agents' forward passes.
    - Configurable rendering, sleep, logging frequency, and max cycles.
    - Uses torch.inference_mode() for faster inference.
    - Optional quieter output mode.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create environment
    env = build_env(render=render, max_cycles=max_cycles_override)

    # Instantiate network & load weights
    net = Net()
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    if not quiet:
        print(f"Testing model: {model_path}")
        print(f"Device: {device}")
        print(f"Episodes: {num_episodes}, Max cycles: {env.max_cycles}, Render: {render}")
        print(f"Batch action: {batch_action}, Sleep per step: {step_sleep}s")

    success_count = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        t_episode_start = time.time()
        obs, _ = env.reset()
        episode_reward = 0.0
        step_count = 0
        episode_success = False

        states = [obs[agent_name] for agent_name in env.agents]

        if not quiet:
            print(f"\nEpisode {episode + 1}")
            print(f"Initial evader position: {env.evader.pos}")

        while step_count < env.max_cycles:
            actions = {}

            # Determine actions (batched or per-agent)
            if batch_action:
                active_indices = [
                    i for i, ag in enumerate(env.agents_objects) if not ag.reached_evader
                ]
                if active_indices:
                    state_tensor = torch.tensor(
                        [states[i] for i in active_indices],
                        dtype=torch.float32,
                        device=device
                    )
                    with torch.inference_mode():
                        eta_q_batch, balance_q_batch = net(state_tensor)
                        eta_actions = torch.argmax(eta_q_batch, dim=1).tolist()
                        balance_actions = torch.argmax(balance_q_batch, dim=1).tolist()

                    # Map back
                    for local_idx, agent_array_idx in enumerate(active_indices):
                        eta_action = eta_actions[local_idx]
                        balance_action = balance_actions[local_idx]
                        eta_scale = net.eta_options[eta_action]
                        individual_balance = net.balance_options[balance_action]
                        agent_name = env.agents[agent_array_idx]
                        actions[agent_name] = np.array(
                            [eta_scale, individual_balance], dtype=np.float32
                        )

                # Default for agents already done
                for i, agent_name in enumerate(env.agents):
                    if env.agents_objects[i].reached_evader:
                        actions[agent_name] = np.array([2.0, 2000.0], dtype=np.float32)

            else:
                # Original per-agent approach
                for i, agent_name in enumerate(env.agents):
                    if not env.agents_objects[i].reached_evader:
                        state_tensor = torch.tensor(
                            states[i], dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        with torch.inference_mode():
                            eta_q, balance_q = net(state_tensor)
                            eta_action = torch.argmax(eta_q, dim=1).item()
                            balance_action = torch.argmax(balance_q, dim=1).item()

                        eta_scale = net.eta_options[eta_action]
                        individual_balance = net.balance_options[balance_action]
                        actions[agent_name] = np.array(
                            [eta_scale, individual_balance], dtype=np.float32
                        )
                    else:
                        actions[agent_name] = np.array([2.0, 2000.0], dtype=np.float32)

            # Environment step
            obs_next, rewards, terminated, truncated, infos = env.step(actions)
            states = [obs_next[agent_name] for agent_name in env.agents]

            episode_reward += sum(rewards.values())
            step_count += 1

            # Periodic logging
            if (not quiet) and log_interval > 0 and (step_count % log_interval == 0):
                active_agents = sum(
                    1 for agent in env.agents_objects if not agent.reached_evader
                )
                print(
                    f"  Step {step_count}: Active agents: {active_agents}, "
                    f"Total reward: {episode_reward:.2f}"
                )

            # Optional rendering
            if render:
                env.render()
            if step_sleep > 0.0:
                time.sleep(step_sleep)

            # Termination check
            if all(terminated.values()) or all(truncated.values()):
                episode_success = any(agent.reached_evader for agent in env.agents_objects)
                break

        # Episode summary
        if episode_success:
            success_count += 1
            if not quiet:
                print(f"  SUCCESS! Episode ended in {step_count} steps")
        else:
            if not quiet:
                print(f"  Failed. Episode ended in {step_count} steps")

        successful_agents = sum(1 for agent in env.agents_objects if agent.reached_evader)
        if not quiet:
            print(f"  Agents that reached evader: {successful_agents}/{env.n_agents}")
            print(f"  Total episode reward: {episode_reward:.2f}")
            print(f"  Episode wall time: {time.time() - t_episode_start:.2f}s")

        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)

    env.close()

    # Final statistics
    if not quiet:
        print(f"\n{'='*50}")
        print(f"TEST RESULTS ({num_episodes} episodes)")
        print(f"{'='*50}")
        print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
        print(
            f"Average episode reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}"
        )
        print(
            f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
        )
        print(f"Best episode reward: {np.max(total_rewards):.2f}")
        print(f"Worst episode reward: {np.min(total_rewards):.2f}")

    # Return metrics if caller wants to programmatically use them
    return {
        "success_rate": success_count / num_episodes,
        "rewards": total_rewards,
        "lengths": episode_lengths,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained multi-agent DQN model.")
    parser.add_argument(
        "model_path",
        nargs="?",
        default="models/dqn_final.pt",
        help="Path to the trained model .pt file",
    )
    parser.add_argument(
        "--episodes", "-e", type=int, default=5, help="Number of test episodes"
    )
    parser.add_argument(
        "--render", action="store_true", help="Enable environment rendering"
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Override environment max cycles (<= original 1000)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds per step (use small value only if rendering)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=250,
        help="Steps between progress prints (0 to disable mid-episode logs)",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batched forward pass over agents (debug mode)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only final summary)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    test_model(
        model_path=args.model_path,
        num_episodes=args.episodes,
        render=args.render,
        max_cycles_override=args.max_cycles,
        step_sleep=args.sleep,
        log_interval=args.log_interval,
        batch_action=not args.no_batch,
        quiet=args.quiet,
    )
