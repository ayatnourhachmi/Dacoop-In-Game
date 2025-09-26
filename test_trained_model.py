import torch
import numpy as np
from train import Net, DQN
from env import ContinuousWorldParallelEnv
import time

def test_model(model_path, num_episodes=10, render=True):
    """Test a trained DQN model"""
    
    # Setup environment
    env = ContinuousWorldParallelEnv(
        n_agents=2, 
        max_cycles=1000, 
        world_size=(10, 10),
        render_mode="human" if render else None,
        dc=0.5, 
        ds=2.0, 
        evader_speed=400, 
        scale=100,
        pursuer_speed=300
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create network and load weights
    net = Net()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    
    print(f"Testing model: {model_path}")
    print(f"Device: {device}")
    
    success_count = 0
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        episode_success = False
        
        states = [obs[agent_name] for agent_name in env.agents]
        
        print(f"\nEpisode {episode + 1}")
        print(f"Initial evader position: {env.evader.pos}")
        
        while step_count < env.max_cycles:
            # Choose actions using trained network
            actions = {}
            
            for i, agent_name in enumerate(env.agents):
                if not env.agents_objects[i].reached_evader:
                    state_tensor = torch.tensor(states[i], dtype=torch.float32, device=device).unsqueeze(0)
                    
                    with torch.no_grad():
                        eta_q, balance_q = net(state_tensor)
                        eta_action = torch.argmax(eta_q, dim=1).item()
                        balance_action = torch.argmax(balance_q, dim=1).item()
                    
                    # Convert to continuous parameters
                    eta_scale = net.eta_options[eta_action]
                    individual_balance = net.balance_options[balance_action]
                    
                    actions[agent_name] = np.array([eta_scale, individual_balance], dtype=np.float32)
                else:
                    actions[agent_name] = np.array([2.0, 2000.0], dtype=np.float32)
            
            # Step environment
            obs_next, rewards, terminated, truncated, infos = env.step(actions)
            states = [obs_next[agent_name] for agent_name in env.agents]
            
            episode_reward += sum(rewards.values())
            step_count += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                active_agents = sum(1 for agent in env.agents_objects if not agent.reached_evader)
                print(f"  Step {step_count}: Active agents: {active_agents}, Total reward: {episode_reward:.2f}")
            
            if render:
                env.render()
                time.sleep(0.05)  # Slow down for visualization
            
            # Check termination
            if all(terminated.values()) or all(truncated.values()):
                episode_success = any(agent.reached_evader for agent in env.agents_objects)
                break
        
        # Episode summary
        if episode_success:
            success_count += 1
            print(f"  SUCCESS! Episode ended in {step_count} steps")
        else:
            print(f"  Failed. Episode ended in {step_count} steps")
        
        successful_agents = sum(1 for agent in env.agents_objects if agent.reached_evader)
        print(f"  Agents that reached evader: {successful_agents}/{env.n_agents}")
        print(f"  Total episode reward: {episode_reward:.2f}")
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
    
    env.close()
    
    # Final statistics
    print(f"\n{'='*50}")
    print(f"TEST RESULTS ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Average episode reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Best episode reward: {np.max(total_rewards):.2f}")
    print(f"Worst episode reward: {np.min(total_rewards):.2f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default to final model
        model_path = "models/dqn_final.pt"
    
    # Test with visualization
    test_model(model_path, num_episodes=5, render=True)
