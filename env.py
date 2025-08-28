from pettingzoo import ParallelEnv
import numpy as np
import random
from renderer import Renderer, Obstacle
from agent import Agent
from evader import Evader

class ContinuousWorldParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "continuous_world_parallel_env_v1"}

    def __init__(self, n_agents=2, max_cycles=1000, world_size=(10, 10),
                 render_mode="human", dc=0.5, ds=2.0, evader_speed=400, scale=100,
                 pursuer_speed=300):
        super().__init__()
        self.n_agents = n_agents
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.possible_agents = self.agents[:]
        self.max_cycles = max_cycles
        self.steps = 0
        self.world_width, self.world_height = world_size
        self.render_mode = render_mode
        self.dt = 0.05  # Smaller timestep for smoother motion
        self.dc = dc
        self.ds = ds
        self.evader_speed = evader_speed  # mm/s like original
        self.pursuer_speed = pursuer_speed  # mm/s, ensure evader_speed > pursuer_speed
        self.scale = scale

        # Scale world coordinates to match original environment (mm)
        self.world_width_mm = self.world_width * 1000
        self.world_height_mm = self.world_height * 1000

        # Obstacle specifications (scaled to match world)
        self.obstacle_specs = [
            (2.0, 2.0, 1.0, 0.5, "sprites/obs1.png"),
            (5.0, 4.0, 0.7, 0.7, "sprites/obs1.png"),
            (7.5, 8.0, 1.2, 0.4, "sprites/obs1.png")
        ]
        self.obstacles_objects = [Obstacle(x, y, w, h, self.scale, image=img)
                                  for x, y, w, h, img in self.obstacle_specs]

        # Agents with APF parameters
        # Convert mm/s to m/s for pursuers constant speed
        pursuer_speed_ms = max(0.01, self.pursuer_speed) / 1000.0
        self.agents_objects = [
            Agent(pos=[1.0 + i * 0.5, 1.0 + i * 0.3],
                  radius=0.3, dc=dc, ds=ds, bounds=world_size,
                  constant_speed=pursuer_speed_ms)
            for i in range(n_agents)
        ]

        # Evader with movement capability
        self.evader = Evader(pos=[6.0, 6.0], scale=self.scale, radius=0.4, de=0.4,
                            image="sprites/evader.png")
        # Evader speed in m/s
        self.evader_speed = max((self.pursuer_speed + 1), evader_speed) / 1000.0
        
        # True for Manual control mode for evader
        self.evader_manual = False

        if render_mode == "human":
            self.renderer = Renderer(
                width_m=self.world_width,
                height_m=self.world_height,
                scale=self.scale,
                agent_radius=0.2,
                obstacles=self.obstacles_objects,
                agent_image="sprites/airplane.png",
                background_image="sprites/bluesky.png"
            )
        else:
            self.renderer = None

        import gymnasium.spaces as gym
        
        # Action space: [eta_scale, individual_balance]
        # eta_scale: 0.1-10.0 (multiplier for repulsive force)
        # individual_balance: 0.0-4000.0 (balance between attraction and repulsion)
        self.action_spaces = {
            a: gym.Box(low=np.array([0.1, 0.0]), high=np.array([10.0, 4000.0]),
                       shape=(2,), dtype=np.float32) for a in self.agents
        }

        # Observation space: enhanced with APF-relevant info
        max_neighbor_info = (n_agents - 1) * 4
        obs_dim = 7 + max_neighbor_info  # pos(2) + evader_pos(2) + nearest_obs(3) + neighbors
        self.observation_spaces = {
            a: gym.Box(-1, max(self.world_width, self.world_height),
                       shape=(obs_dim,), dtype=np.float32) for a in self.agents
        }

    def print_detailed_agent_info(self):
        """Print comprehensive information about each agent"""
        print("\n" + "="*80)
        print(f"STEP {self.steps} - DETAILED AGENT INFORMATION")
        print("="*80)
        
        for i, agent in enumerate(self.agents_objects):
            agent_name = self.agents[i]
            
            # Basic agent info
            dist_to_evader = np.linalg.norm(agent.pos - self.evader.pos)
            status = "TERMINATED" if agent.reached_evader else "ACTIVE"
            wall_status = "WALL_FOLLOWING" if agent.wall_following else "NORMAL"
            
            print(f"\n{agent_name.upper()} [{status}] [{wall_status}]")
            print(f"   Position: ({agent.pos[0]:.3f}, {agent.pos[1]:.3f})")
            print(f"   Orientation: ({agent.orientation[0]:.3f}, {agent.orientation[1]:.3f})")
            print(f"   Distance to Evader: {dist_to_evader:.3f}")
            print(f"   Reached Evader: {'TRUE' if agent.reached_evader else 'FALSE'}")
            
            # Get neighbors information
            neighbors_info = agent.get_neighbors_info(self.agents_objects, self.evader.pos)
            print(f"   Neighbors within ds={agent.ds:.1f}: {len(neighbors_info)}")
            
            if neighbors_info:
                for j, neighbor in enumerate(neighbors_info):
                    neighbor_name = f"agent_{neighbor['agent_id']}"
                    neighbor_status = "ACTIVE" if neighbor['is_active'] else "TERMINATED"
                    print(f"      [{j+1}] {neighbor_name} [{neighbor_status}]")
                    print(f"          Position: ({neighbor['position'][0]:.3f}, {neighbor['position'][1]:.3f})")
                    print(f"          Distance to me: {neighbor['distance_to_me']:.3f}")
                    print(f"          Distance to evader: {neighbor['distance_to_evader']:.3f}")
            else:
                print("      No neighbors detected")
            
            # Get nearest obstacle information
            nearest_obstacle = agent.get_nearest_obstacle_info(self.obstacles_objects)
            if nearest_obstacle:
                print(f"   Nearest Obstacle:")
                print(f"      Distance: {nearest_obstacle['distance']:.3f}")
                print(f"      Closest point: ({nearest_obstacle['position'][0]:.3f}, {nearest_obstacle['position'][1]:.3f})")
            else:
                print("   Nearest Obstacle: None detected")
            
            # Check collision with obstacles
            collision_with_obstacle = any(
                obs.collides_with_point(agent.pos, agent.dc)
                for obs in self.obstacles_objects
            )
            print(f"   Near Obstacle (collision): {'TRUE' if collision_with_obstacle else 'FALSE'}")
        
        # Evader info
        evader_info = self.evader.get_status_info()
        print(f"\nEVADER INFORMATION [{evader_info['status']}]")
        print(f"   Position: ({evader_info['position'][0]:.3f}, {evader_info['position'][1]:.3f})")
        print(f"   Direction: ({evader_info['direction'][0]:.3f}, {evader_info['direction'][1]:.3f})")
        print(f"   Visual Radius: {evader_info['visual_radius']:.3f}")
        print(f"   Collision Radius (de): {evader_info['collision_radius']:.3f}")
        
        print("\n" + "="*80)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.steps = 0

        # Reset agent positions safely
        for i, agent in enumerate(self.agents_objects):
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                x = random.uniform(agent.radius, self.world_width - agent.radius)
                y = random.uniform(agent.radius, self.world_height - agent.radius)
                collision = any(obs.collides_with_point((x, y), agent.dc)
                                for obs in self.obstacles_objects)
                collision = collision or any(
                    np.linalg.norm(np.array([x, y]) - np.array(other.pos)) < agent.dc * 2
                    for j, other in enumerate(self.agents_objects) if j < i
                )
                if not collision:
                    agent.pos[:] = [x, y]
                    placed = True
                attempts += 1
            if not placed:
                agent.pos[:] = [1.0 + i * 0.5, 1.0 + i * 0.3]
            
            # Reset agent state
            agent.reached_evader = False
            agent.last_velocity = np.zeros(2)
            agent.wall_following = False
            agent.orientation = np.array([0.0, 1.0])  # Reset orientation

        # Reset evader position using the new Evader class method
        clearance = max(self.evader.de * 2.0, 0.6)
        self.evader.set_random_position(self.world_width, self.world_height, 
                                       self.obstacles_objects, clearance)
        
        # Reset evader state
        self.evader.reset_state()

        obs = {a: self._get_obs(i) for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}
        
        # Print initial state
        self.print_detailed_agent_info()
        
        return obs, infos

    def _get_obs(self, agent_idx):
        """Get enhanced observation for a specific agent"""
        agent = self.agents_objects[agent_idx]
        
        # Basic info: agent position + evader position
        obs = list(agent.pos) + list(self.evader.pos)
        
        # Nearest obstacle info (distance, position)
        nearest_obstacle = agent.get_nearest_obstacle_info(self.obstacles_objects)
        if nearest_obstacle:
            obs.extend([
                nearest_obstacle['distance'],
                nearest_obstacle['position'][0],
                nearest_obstacle['position'][1]
            ])
        else:
            obs.extend([-1.0, -1.0, -1.0])  # No obstacle detected
        
        # Neighbors info
        neighbors_info = agent.get_neighbors_info(self.agents_objects, self.evader.pos)
        max_neighbors = self.n_agents - 1
        
        # Add neighbor information (pad with -1 if fewer neighbors)
        for i in range(max_neighbors):
            if i < len(neighbors_info):
                neighbor = neighbors_info[i]
                obs.extend([
                    neighbor['position'][0],
                    neighbor['position'][1],
                    neighbor['distance_to_evader'],
                    1.0 if neighbor['is_active'] else 0.0
                ])
            else:
                obs.extend([-1.0, -1.0, -1.0, -1.0])  # No neighbor
        
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        self.steps += 1
        total_rewards = {}
        terminated = {}
        truncated = {}
        infos = {}

        # Update agents with their actions
        for i, agent in enumerate(self.agents_objects):
            agent_name = self.agents[i]
            action = actions[agent_name]
            
            if not agent.reached_evader:
                agent.update(
                    dt=self.dt,
                    bounds=(self.world_width, self.world_height),
                    obstacles=self.obstacles_objects,
                    agents=self.agents_objects,
                    evader_pos=self.evader.pos,
                    evader_de=self.evader.de,
                    action=action,  # Pass the APF parameters
                    eta=1.0,
                    rho_0=3.0,
                    lam=1.0,
                    repulse_gain=2.0,
                    wall_rho0=2.0,
                    wall_repulse_gain=2.0
                )

        # Calculate rewards and termination conditions
        for i, agent in enumerate(self.agents_objects):
            agent_name = self.agents[i]
            
            # Calculate distance to evader
            dist_to_evader_center = np.linalg.norm(agent.pos - self.evader.pos)
            dist_to_evader_surface = max(0, dist_to_evader_center - self.evader.de)
            
            # Reward structure similar to original environment
            reward = 0.0
            # Persist termination once reached
            done_flag = agent.reached_evader
            # Ensure variable is always defined for infos below
            collision_with_obstacle = False
            
            if not agent.reached_evader:
                # Main reward - capture bonus
                if dist_to_evader_surface <= agent.dc:
                    reward += 20.0  # r_main
                    done_flag = True
                    agent.reached_evader = True
                
                # Collision penalties
                # Obstacle collision penalty
                collision_with_obstacle = any(
                    obs.collides_with_point(agent.pos, 0.1)  # 100mm like original
                    for obs in self.obstacles_objects
                )
                if collision_with_obstacle:
                    reward -= 20.0  # r_col_1
                elif any(obs.collides_with_point(agent.pos, 0.15) for obs in self.obstacles_objects):
                    reward -= 2.0  # Warning zone
                
                # Agent collision penalty
                min_agent_dist = float('inf')
                for other in self.agents_objects:
                    if other is not agent:
                        other_dist = np.linalg.norm(agent.pos - other.pos)
                        min_agent_dist = min(min_agent_dist, other_dist)
                
                if min_agent_dist < 0.2:  # 200mm like original
                    reward -= 20.0  # r_col_2
                
                # Approach reward - distance-based improvement
                current_dist = dist_to_evader_center
                if hasattr(agent, 'last_distance_to_evader'):
                    distance_improvement = agent.last_distance_to_evader - current_dist
                    reward += distance_improvement / 0.2  # r_app, normalized by 200mm
                agent.last_distance_to_evader = current_dist
                
                # Movement reward (encourage active behavior)
                reward += 0.1 * np.linalg.norm(agent.last_velocity)
            
            # Termination conditions
            if self.steps >= self.max_cycles:
                done_flag = True  # Time limit reached
            
            # Store results
            total_rewards[agent_name] = reward
            terminated[agent_name] = done_flag
            truncated[agent_name] = False
            
            # Information for debugging
            neighbors_info = agent.get_neighbors_info(self.agents_objects, self.evader.pos)
            nearest_obstacle = agent.get_nearest_obstacle_info(self.obstacles_objects)
            
            infos[agent_name] = {
                "distance_to_evader": dist_to_evader_center,
                "distance_to_evader_surface": dist_to_evader_surface,
                "near_obstacle": collision_with_obstacle,
                "reached_evader": agent.reached_evader,
                "terminated": done_flag,
                "reward": reward,
                "neighbors_count": len(neighbors_info),
                "neighbors_info": neighbors_info,
                "nearest_obstacle": nearest_obstacle,
                "position": agent.pos.copy(),
                "orientation": agent.orientation.copy(),
                "wall_following": agent.wall_following,
                "status": "TERMINATED" if agent.reached_evader else "ACTIVE",
                "action_used": actions[agent_name].copy()
            }

        # Update evader position only if no agent has captured it
        if not any(agent.reached_evader for agent in self.agents_objects):
            if self.evader_manual:
                # Manual control mode
                self.evader.update_manual_control(
                    speed=self.evader_speed,
                    dt=self.dt,
                    world_bounds=(self.world_width, self.world_height),
                    obstacles=self.obstacles_objects
                )
            else:
                # Autonomous mode
                self.evader.update_autonomous(
                    agents=self.agents_objects,
                    obstacles=self.obstacles_objects,
                    world_bounds=(self.world_width, self.world_height),
                    speed=self.evader_speed,
                    dt=self.dt
                )

        # Print detailed information periodically
        if self.steps % 30 == 0 or any(agent.reached_evader for agent in self.agents_objects):
            self.print_detailed_agent_info()

        obs = {a: self._get_obs(i) for i, a in enumerate(self.agents)}
        return obs, total_rewards, terminated, truncated, infos

    def render(self):
        if self.renderer:
            agent_positions = [agent.pos for agent in self.agents_objects]
            self.renderer.draw(agent_positions, self.evader, agents=self.agents_objects)
            self.renderer.tick(60)

    def close(self):
        if self.renderer:
            self.renderer.close()

if __name__ == "__main__":
    # Test with APF actions
    print("=== APF-Based Navigation Test ===")
    env = ContinuousWorldParallelEnv(n_agents=2, render_mode="human")
    num_episodes = 2

    for episode in range(num_episodes):
        obs, infos = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        print(f"\nEpisode {episode + 1} started")
        print(f"Evader position: {env.evader.pos}")

        while not done and step_count < 1000:
            # Create APF-based actions
            actions = {}
            for agent_name in env.agents:
                # eta_scale: moderate repulsion, individual_balance: balanced
                actions[agent_name] = np.array([2.0, 2000.0], dtype=np.float32)
            
            obs, rewards, terminated, truncated, infos = env.step(actions)

            total_reward += sum(rewards.values())
            step_count += 1

            if step_count % 100 == 0:
                print(f"Step {step_count}: Total reward={total_reward:.2f}")
                for agent_name, info in infos.items():
                    print(f"  {agent_name}: pos=({info['position'][0]:.2f},{info['position'][1]:.2f}), "
                          f"dist_to_evader={info['distance_to_evader']:.2f}, "
                          f"wall_following={info['wall_following']}")

            env.render()
            done = all(terminated.values()) or any(truncated.values())

        print(f"Episode {episode + 1} finished in {step_count} steps. Total reward: {total_reward:.3f}")
        success_count = sum(1 for agent in env.agents_objects if agent.reached_evader)
        print(f"Agents that reached evader: {success_count}/{env.n_agents}")

    env.close()