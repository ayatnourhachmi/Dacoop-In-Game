import torch
import numpy as np
import argparse
import time
from env import ContinuousWorldParallelEnv
from model_def import Net

def make_env(render=False, max_cycles=1000):
    return ContinuousWorldParallelEnv(
        n_agents=2,
        max_cycles=max_cycles,
        world_size=(10, 10),
        render_mode="human" if render else None,
        dc=0.5,
        ds=2.0,
        evader_speed=400,
        scale=100,
        pursuer_speed=300
    )

@torch.inference_mode()
def select_actions(net, device, states, agents_done_mask):
    """
    Batched action selection for all active agents.
    states: list of per-agent state vectors (np arrays).
    agents_done_mask: list[bool] True if agent already reached evader.
    Returns dict agent_index -> np.array([eta_scale, balance])
    """
    actions = {}
    active_indices = [i for i, done in enumerate(agents_done_mask) if not done]
    if active_indices:
        batch = torch.tensor([states[i] for i in active_indices],
                             dtype=torch.float32, device=device)
        eta_q, bal_q = net(batch)
        eta_actions = eta_q.argmax(1).tolist()
        bal_actions = bal_q.argmax(1).tolist()
        for pos, agent_idx in enumerate(active_indices):
            eta_scale = net.eta_options[eta_actions[pos]]
            balance = net.balance_options[bal_actions[pos]]
            actions[agent_idx] = np.array([eta_scale, balance], dtype=np.float32)
    # Fill defaults for done agents
    for i, done in enumerate(agents_done_mask):
        if done:
            actions[i] = np.array([2.0, 2000.0], dtype=np.float32)
    return actions

def evaluate(
    model_path: str,
    episodes: int = 10,
    render: bool = False,
    max_cycles: int = 1000,
    sleep: float = 0.0,
    log_interval: int = 250,
    quiet: bool = False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_env(render=render, max_cycles=max_cycles)

    # Derive state dimension dynamically
    obs, _ = env.reset()
    first_state = obs[env.agents[0]]
    num_state = len(first_state)

    net = Net(num_state)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    if not quiet:
        print(f"Loaded model: {model_path}")
        print(f"Device: {device}, Episodes: {episodes}, Max cycles: {env.max_cycles}")

    rewards_per_episode = []
    lengths = []
    success_flags = []
    capture_steps = []  # steps taken in successful episodes

    for ep in range(episodes):
        ep_reward = 0.0
        obs, _ = env.reset()
        states = [obs[name] for name in env.agents]
        step = 0
        success = False

        if not quiet:
            print(f"\nEpisode {ep+1}")

        while step < env.max_cycles:
            agents_done_mask = [ag.reached_evader for ag in env.agents_objects]
            actions_indexed = select_actions(net, device, states, agents_done_mask)

            # Map index->action to env action dict
            action_dict = {env.agents[i]: actions_indexed[i] for i in range(len(env.agents))}
            obs_next, rewards, terminated, truncated, infos = env.step(action_dict)
            states = [obs_next[name] for name in env.agents]

            ep_reward += sum(rewards.values())
            step += 1

            if log_interval > 0 and (step % log_interval == 0) and (not quiet):
                active = sum(1 for a in env.agents_objects if not a.reached_evader)
                print(f"  Step {step}: cumulative reward {ep_reward:.2f}, active agents {active}")

            if render:
                env.render()
            if sleep > 0:
                time.sleep(sleep)

            if all(terminated.values()) or all(truncated.values()):
                success = any(a.reached_evader for a in env.agents_objects)
                if success:
                    capture_steps.append(step)
                break

        rewards_per_episode.append(ep_reward)
        lengths.append(step)
        success_flags.append(1 if success else 0)

        if not quiet:
            if success:
                print(f"  SUCCESS in {step} steps, reward={ep_reward:.2f}")
            else:
                print(f"  FAIL (maxed out or truncated), reward={ep_reward:.2f}")

    env.close()

    success_rate = np.mean(success_flags) if success_flags else 0.0
    avg_len = np.mean(lengths)
    avg_reward = np.mean(rewards_per_episode)
    if not quiet:
        print("\n================= EVALUATION SUMMARY =================")
        print(f"Episodes: {episodes}")
        print(f"Success Rate: {success_rate*100:.1f}%")
        print(f"Average Reward: {avg_reward:.2f}  (std {np.std(rewards_per_episode):.2f})")
        print(f"Average Length: {avg_len:.1f}  (std {np.std(lengths):.1f})")
        if capture_steps:
            print(f"Avg Capture Steps (successful eps): {np.mean(capture_steps):.1f}")
        print("Best Reward:", np.max(rewards_per_episode))
        print("Worst Reward:", np.min(rewards_per_episode))

    return {
        "success_rate": success_rate,
        "rewards": rewards_per_episode,
        "lengths": lengths,
        "capture_steps": capture_steps
    }

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="models/dqn_final.pt")
    ap.add_argument("--episodes", "-e", type=int, default=5)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--max-cycles", type=int, default=1000)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--log-interval", type=int, default=250)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        render=args.render,
        max_cycles=args.max_cycles,
        sleep=args.sleep,
        log_interval=args.log_interval,
        quiet=args.quiet
    )