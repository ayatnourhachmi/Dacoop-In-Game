from pettingzoo import ParallelEnv
import numpy as np
import random
from renderer import Renderer, Evader, Obstacle
from agent import Agent

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
        self.evader_direction = np.array([0.0, 1.0])  # Initial direction
        # Maintain a minimum clearance from walls (in meters)
        self.evader_clearance = max(self.evader.de * 2.0, 0.6)
        # Manual control for evader and corner var init
        self.evader_manual = True
        self._corner_mode_steps = 0
        
        # Evader behavior state
        self.evader_wall_following = False
        self.evader_zigzag_flag = False
        self.evader_slip_flag = False
        self.zigzag_count = 0
        self.zigzag_last = np.zeros(2)
        self.last_e = np.zeros(2)

        # Manual control mode for evader
        self.evader_manual = True

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
        evader_status = "WALL_FOLLOWING" if self.evader_wall_following else "NORMAL"
        if self.evader_zigzag_flag:
            evader_status = "ZIGZAG"
        if self.evader_slip_flag:
            evader_status = "SLIP"
        
        print(f"\nEVADER INFORMATION [{evader_status}]")
        print(f"   Position: ({self.evader.pos[0]:.3f}, {self.evader.pos[1]:.3f})")
        print(f"   Direction: ({self.evader_direction[0]:.3f}, {self.evader_direction[1]:.3f})")
        print(f"   Visual Radius: {self.evader.radius:.3f}")
        print(f"   Collision Radius (de): {self.evader.de:.3f}")
        
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

        # Reset evader position (like original environment)
        evader_placed = False
        attempts = 0
        while not evader_placed and attempts < 100:
            x = random.uniform(self.evader_clearance, self.world_width - self.evader_clearance)
            y = random.uniform(max(self.evader_clearance, self.world_height * 0.7), self.world_height - self.evader_clearance)  # keep clearance from walls
            collision = any(obs.collides_with_point((x, y), self.evader.de)
                            for obs in self.obstacles_objects)
            if not collision:
                self.evader.pos[:] = [x, y]
                evader_placed = True
            attempts += 1
        if not evader_placed:
            self.evader.pos[:] = [self.world_width / 2, self.world_height * 0.8]

        # Reset evader state
        self.evader_direction = np.array([0.0, 1.0])
        self.evader_wall_following = False
        self.evader_zigzag_flag = False
        self.evader_slip_flag = False
        self.zigzag_count = 0
        self.zigzag_last = np.zeros(2)
        self.last_e = np.zeros(2)

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

    def _compute_evader_force(self):
        """Compute desired evader unit direction using zigzag/escape logic (no repulsive forces)."""
        # Corner detection
        near_left = self.evader.pos[0] <= self.evader_clearance + 1e-6
        near_right = self.evader.pos[0] >= self.world_width - self.evader_clearance - 1e-6
        near_top = self.evader.pos[1] <= self.evader_clearance + 1e-6
        near_bottom = self.evader.pos[1] >= self.world_height - self.evader_clearance - 1e-6
        near_two_walls = (near_left or near_right) and (near_top or near_bottom)

        if near_two_walls or self._corner_mode_steps > 0:
            # Compute wall normal and tangential direction; persist for a few frames
            wall_normal = np.array([0.0, 0.0])
            if near_left:
                wall_normal += np.array([1.0, 0.0])
            if near_right:
                wall_normal += np.array([-1.0, 0.0])
            if near_top:
                wall_normal += np.array([0.0, 1.0])
            if near_bottom:
                wall_normal += np.array([0.0, -1.0])
            if np.linalg.norm(wall_normal) < 1e-6:
                wall_normal = -self.evader_direction
            wall_normal = wall_normal / (np.linalg.norm(wall_normal) + 1e-9)
            t1 = np.array([-wall_normal[1], wall_normal[0]])
            t2 = -t1
            # Prefer tangent aligned with current direction to avoid jitter
            cand = t1 if np.dot(t1, self.evader_direction) >= np.dot(t2, self.evader_direction) else t2
            # Persist choice
            self._corner_tangent = cand
            self._corner_mode_steps = max(self._corner_mode_steps, 12)
            self.evader_wall_following = True
            return cand / (np.linalg.norm(cand) + 1e-9)
        # Distances to agents
        agent_positions = [a.pos for a in self.agents_objects if not a.reached_evader]
        if len(agent_positions) == 0:
            mean_escape = np.zeros(2)
            nearest_agent_dist = float('inf')
        else:
            diffs = [self.evader.pos - ap for ap in agent_positions]
            dists = [np.linalg.norm(d) for d in diffs]
            nearest_agent_dist = min(dists)
            # Weighted average of away vectors (inverse-square like)
            contribs = []
            for d, r in zip(diffs, dists):
                if r > 1e-6:
                    contribs.append(d / (r * r))
            mean_escape = np.mean(contribs, axis=0) if contribs else np.zeros(2)

        # Panic level
        if nearest_agent_dist == float('inf') or nearest_agent_dist > self.evader_sensitivity_range:
            p_panic = 0.0
        else:
            x = -nearest_agent_dist / self.evader_sensitivity_range + 1.0
            p_panic = (np.exp(x) - 1.0) / (np.e - 1.0)

        # Zigzag if not panicked and no close obstacle
        # Compute min distance to obstacles (via closest point on rectangle)
        def dist_to_obstacle(obs):
            left = obs.x - obs.width_m / 2
            right = obs.x + obs.width_m / 2
            top = obs.y - obs.height_m / 2
            bottom = obs.y + obs.height_m / 2
            cx = np.clip(self.evader.pos[0], left, right)
            cy = np.clip(self.evader.pos[1], top, bottom)
            return np.linalg.norm(self.evader.pos - np.array([cx, cy]))

        min_obs_dist = min([dist_to_obstacle(o) for o in self.obstacles_objects], default=float('inf'))

        if p_panic < self.evader_zigzag_threshold and min_obs_dist > 0.3:
            self.evader_zigzag_flag = True
            if self.zigzag_count == 0:
                # pick a random direction roughly forward
                while True:
                    ang = np.random.uniform(-np.pi, np.pi)
                    cand = np.array([np.cos(ang), np.sin(ang)])
                    if np.dot(cand, self.evader_direction) > 0:
                        self.zigzag_last = cand
                        break
                self.zigzag_count = 1
                F_escape = self.zigzag_last
            else:
                F_escape = self.zigzag_last
                self.zigzag_count += 1
                if self.zigzag_count > np.random.randint(self.evader_zigzag_min, self.evader_zigzag_max + 1):
                    self.zigzag_count = 0
        else:
            self.evader_zigzag_flag = False
            self.zigzag_count = 0
            # Use averaged away vector
            if np.linalg.norm(mean_escape) > 1e-6:
                F_escape = mean_escape / np.linalg.norm(mean_escape)
            else:
                F_escape = self.evader_direction.copy()

        # Predictive check: if stepping along escape would collide with walls/obstacles, follow wall/obstacle tangent
        step_preview = self.evader_speed * self.dt
        cand_pos = self.evader.pos + F_escape * step_preview

        def would_hit_walls(p):
            return (
                p[0] < self.evader_clearance or p[0] > self.world_width - self.evader_clearance or
                p[1] < self.evader_clearance or p[1] > self.world_height - self.evader_clearance
            )

        def would_hit_obstacles(p):
            return any(obs.collides_with_point(p, self.evader.de) for obs in self.obstacles_objects)

        self.evader_wall_following = False
        if would_hit_walls(cand_pos) or would_hit_obstacles(cand_pos):
            # Choose tangential direction around nearest constraint
            # Prefer nearest obstacle if any
            nearest_o = None
            if len(self.obstacles_objects) > 0:
                dists = [(dist_to_obstacle(o), o) for o in self.obstacles_objects]
                dists.sort(key=lambda x: x[0])
                nearest_o = dists[0][1]
            if nearest_o is not None and dist_to_obstacle(nearest_o) < float('inf'):
                left = nearest_o.x - nearest_o.width_m / 2
                right = nearest_o.x + nearest_o.width_m / 2
                topy = nearest_o.y - nearest_o.height_m / 2
                bottom = nearest_o.y + nearest_o.height_m / 2
                cx = np.clip(self.evader.pos[0], left, right)
                cy = np.clip(self.evader.pos[1], topy, bottom)
                closest = np.array([cx, cy])
                normal = self.evader.pos - closest
            else:
                # Use wall normal
                normal = np.array([0.0, 0.0])
                if self.evader.pos[0] < self.evader_clearance:
                    normal += np.array([1.0, 0.0])
                if self.evader.pos[0] > self.world_width - self.evader_clearance:
                    normal += np.array([-1.0, 0.0])
                if self.evader.pos[1] < self.evader_clearance:
                    normal += np.array([0.0, 1.0])
                if self.evader.pos[1] > self.world_height - self.evader_clearance:
                    normal += np.array([0.0, -1.0])
                if np.linalg.norm(normal) < 1e-6:
                    normal = -F_escape
            if np.linalg.norm(normal) > 1e-6:
                normal = normal / np.linalg.norm(normal)
                tangential = np.array([-normal[1], normal[0]])
                if np.dot(tangential, self.evader_direction) < 0:
                    tangential = -tangential
                F_total = tangential
                self.evader_wall_following = True
            else:
                F_total = F_escape
        else:
            F_total = F_escape

        # Normalize
        if np.linalg.norm(F_total) > 1e-6:
            F_total = F_total / np.linalg.norm(F_total)
        else:
            F_total = self.evader_direction.copy()
        return F_total

    def update_evader(self):
        """Update evader position via manual keyboard control (arrow keys/WASD)."""
        try:
            import pygame
            keys = pygame.key_get_pressed() if hasattr(pygame, 'key_get_pressed') else pygame.key.get_pressed()
            dx = (1 if keys[pygame.K_RIGHT] or keys[pygame.K_d] else 0) - (1 if keys[pygame.K_LEFT] or keys[pygame.K_a] else 0)
            dy = (1 if keys[pygame.K_DOWN] or keys[pygame.K_s] else 0) - (1 if keys[pygame.K_UP] or keys[pygame.K_w] else 0)
        except Exception:
            dx, dy = 0, 0

        move = np.array([dx, dy], dtype=float)
        if np.linalg.norm(move) > 1e-6:
            move_dir = move / np.linalg.norm(move)
            self.evader_direction = move_dir
            step = self.evader_speed * self.dt
            candidate = self.evader.pos + move_dir * step
            # Keep wall clearance
            candidate[0] = np.clip(candidate[0], self.evader_clearance, self.world_width - self.evader_clearance)
            candidate[1] = np.clip(candidate[1], self.evader_clearance, self.world_height - self.evader_clearance)
            # Skip move if collides with obstacles
            if not any(obs.collides_with_point(candidate, self.evader.de) for obs in self.obstacles_objects):
                self.evader.pos[:] = candidate

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

        # Evader update is deferred until after capture checks

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
            self.update_evader()
            # Provide current direction to renderer for sprite rotation
            if hasattr(self, 'renderer') and self.renderer:
                # store direction on evader so renderer can rotate sprite
                self.evader.direction = self.evader_direction.copy()

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