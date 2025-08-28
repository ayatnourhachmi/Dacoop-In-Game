import numpy as np

class Agent:
    def __init__(self, pos, radius=0.3, min_speed=0.4, max_speed=4.0, dc=1.5, ds=3.0, bounds=None):
        self.pos = np.array(pos, dtype=np.float32)
        self.radius = radius  # physical size
        self.dc = dc          # capture distance
        self.ds = ds          # sensing distance
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.hit_wall = False
        self.reached_evader = False
        self.last_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.bounds = bounds
        self.wall_obstacles = self._create_wall_obstacles(self.bounds) if self.bounds else []
        
        # APF parameters
        self.wall_following = False
        self.orientation = np.array([0.0, 1.0], dtype=np.float32)  # Initial heading

    def get_neighbors_info(self, agents, evader_pos):
        """Get information about neighbors within sensing distance ds"""
        neighbors_info = []
        
        for i, other in enumerate(agents):
            if other is self:
                continue
                
            diff = other.pos - self.pos
            dist_to_neighbor = np.linalg.norm(diff)
            
            # Check if neighbor is within sensing distance
            if dist_to_neighbor <= self.ds:
                # Calculate neighbor's distance to evader
                neighbor_evader_dist = np.linalg.norm(other.pos - evader_pos)
                
                neighbor_info = {
                    'agent_id': i,
                    'position': other.pos.copy(),
                    'distance_to_me': dist_to_neighbor,
                    'distance_to_evader': neighbor_evader_dist,
                    'is_active': not other.reached_evader,
                    'reached_evader': other.reached_evader
                }
                neighbors_info.append(neighbor_info)
        
        return neighbors_info

    def get_nearest_obstacle_info(self, obstacles):
        """Get information about the nearest obstacle"""
        if not obstacles:
            return None
            
        min_distance = float('inf')
        nearest_obstacle = None
        nearest_pos = None
        
        for obs in obstacles:
            if hasattr(obs, "width_m") and hasattr(obs, "height_m"):
                # Rectangular obstacle
                left = obs.x - obs.width_m / 2
                right = obs.x + obs.width_m / 2
                top = obs.y - obs.height_m / 2
                bottom = obs.y + obs.height_m / 2
                
                # Find closest point on rectangle
                cx = np.clip(self.pos[0], left, right)
                cy = np.clip(self.pos[1], top, bottom)
                closest_point = np.array([cx, cy])
                
                distance = np.linalg.norm(self.pos - closest_point)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_obstacle = obs
                    nearest_pos = closest_point
            else:
                # Point obstacle or agent
                if isinstance(obs, Agent):
                    obs_pos = obs.pos
                else:
                    obs_pos = np.array([obs.x, obs.y])
                distance = np.linalg.norm(self.pos - obs_pos)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_obstacle = obs
                    nearest_pos = obs_pos
        
        if nearest_obstacle is not None:
            return {
                'distance': min_distance,
                'position': nearest_pos,
                'obstacle_center': np.array([nearest_obstacle.x, nearest_obstacle.y]) if hasattr(nearest_obstacle, 'x') else nearest_pos
            }
        
        return None

    def update(self, dt, bounds, obstacles, evader_pos, evader_de, agents, action=None, eta=1.0, rho_0=2.0, lam=1.0):
        """
        Update agent position using APF with action-based parameters
        """
        # Check if already reached evader
        dist_to_evader = np.linalg.norm(self.pos - evader_pos)
        dist_to_evader_surface = max(0, dist_to_evader - evader_de)
        if dist_to_evader_surface <= self.dc:
            self.reached_evader = True
            return 0.0

        if self.reached_evader:
            return 0.0

        # Use action to determine APF parameters if provided
        if action is not None:
            # Action is 2D: [scale_repulse_factor, balance_factor]
            # Map action[0] to eta (repulsive force scale)
            eta = np.clip(action[0], 0.1, 10.0) * 1e6
            # Map action[1] to individual balance (attractive vs repulsive balance)
            individual_balance = np.clip(action[1], 0.0, 4000.0)
        else:
            individual_balance = 2000.0  # default

        # Calculate APF forces
        F_attractive = self.attractive_force(evader_pos)
        
        # Create obstacle list including walls and other agents
        all_obstacles = obstacles + self.wall_obstacles + [agent for agent in agents if agent is not self]
        F_repulsive = self.repulsive_force(all_obstacles, eta, rho_0)
        
        # Inter-individual force (social coordination)
        F_social = self.inter_individual_force(agents, lam)
        
        # Combine forces with individual balance
        total_force = F_attractive + F_repulsive + F_social
        
        # Check for local minimum (APF wall-following logic)
        self.wall_following = False
        if np.linalg.norm(F_attractive) > 1e-6 and np.linalg.norm(F_repulsive) > 1e-6:
            angle = self.angle_between(F_attractive, F_repulsive)
            if angle > np.pi / 2:  # Forces opposing each other
                self.wall_following = True
                total_force = self.wall_following_force(F_attractive, F_repulsive, F_social)

        # Apply angular constraint (max 30 degrees turn per step)
        if np.linalg.norm(total_force) > 1e-6:
            desired_direction = total_force / np.linalg.norm(total_force)
            max_turn_angle = np.radians(30)
            
            current_angle = np.arctan2(self.orientation[1], self.orientation[0])
            desired_angle = np.arctan2(desired_direction[1], desired_direction[0])
            
            angle_diff = desired_angle - current_angle
            # Normalize angle difference to [-pi, pi]
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # Limit turn rate
            actual_angle_diff = np.clip(angle_diff, -max_turn_angle, max_turn_angle)
            actual_angle = current_angle + actual_angle_diff
            
            # Update orientation
            self.orientation = np.array([np.cos(actual_angle), np.sin(actual_angle)])
            
            # Calculate velocity
            speed = np.clip(np.linalg.norm(total_force), self.min_speed, self.max_speed)
            velocity = self.orientation * speed
        else:
            velocity = np.zeros(2, dtype=np.float32)

        # Update position
        self.pos = self.pos + velocity * dt
        self.last_velocity = velocity.copy()

        return np.linalg.norm(velocity)

    def wall_following_force(self, F_attractive, F_repulsive, F_social):
        """Wall following strategy when in local minimum"""
        if np.linalg.norm(F_repulsive) < 1e-6:
            return F_attractive + F_social
            
        # Get perpendicular directions to repulsive force
        F_rep_unit = F_repulsive / np.linalg.norm(F_repulsive)
        perp1 = np.array([-F_rep_unit[1], F_rep_unit[0]])
        perp2 = np.array([F_rep_unit[1], -F_rep_unit[0]])
        
        # Choose direction that better aligns with attractive force
        if np.linalg.norm(F_attractive) > 1e-6:
            F_att_unit = F_attractive / np.linalg.norm(F_attractive)
            align1 = np.dot(perp1, F_att_unit)
            align2 = np.dot(perp2, F_att_unit)
            chosen_direction = perp1 if align1 > align2 else perp2
        else:
            chosen_direction = perp1
        
        # Scale wall following force
        wall_force_magnitude = np.linalg.norm(F_attractive) * 0.8
        return chosen_direction * wall_force_magnitude + F_social

    def inter_individual_force(self, agents, lam=1.0):
        """Social coordination force"""
        force = np.zeros(2, dtype=np.float32)
        count = 0
        
        for other in agents:
            if other is self or other.reached_evader:
                continue
                
            diff = other.pos - self.pos
            dist = np.linalg.norm(diff)
            
            if 1e-6 < dist <= self.ds:
                # Alignment: match neighbor velocities
                vel_diff = other.last_velocity - self.last_velocity
                alignment_force = 0.3 * vel_diff
                
                # Cohesion: move toward neighbors
                cohesion_force = 0.2 * (diff / dist)
                
                # Separation: avoid crowding
                if dist < self.ds * 0.5:
                    separation_force = -0.4 * (diff / dist) / dist
                else:
                    separation_force = np.zeros(2, dtype=np.float32)
                
                force += alignment_force + cohesion_force + separation_force
                count += 1
        
        if count > 0:
            force /= count
            
        return lam * force

    def attractive_force(self, evader_pos):
        """Attractive force toward evader"""
        evader = np.array(evader_pos, dtype=np.float32)
        diff = evader - self.pos
        dist = np.linalg.norm(diff)

        if dist < 1e-6:
            return np.zeros(2, dtype=np.float32)

        # Distance-based scaling
        if dist > 5.0:
            scale = 2.0
        elif dist > 2.0:
            scale = 1.5
        else:
            scale = 0.5
            
        return scale * (diff / dist)

    def repulsive_force(self, obstacles, eta=1.0, rho_0=2.0):
        """Repulsive force from obstacles"""
        force = np.zeros(2, dtype=np.float32)
        px, py = self.pos

        for obs in obstacles:
            # Calculate distance and direction
            if hasattr(obs, "width_m") and hasattr(obs, "height_m"):
                # Rectangular obstacle
                left = obs.x - obs.width_m / 2
                right = obs.x + obs.width_m / 2
                top = obs.y - obs.height_m / 2
                bottom = obs.y + obs.height_m / 2
                cx = np.clip(px, left, right)
                cy = np.clip(py, top, bottom)
                closest = np.array([cx, cy], dtype=np.float32)
                diff = self.pos - closest
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    diff = np.array([1.0, 0.0], dtype=np.float32)
                    dist = 1e-6
            else:
                # Agent or point obstacle
                if isinstance(obs, Agent):
                    obs_pos = obs.pos
                else:
                    obs_pos = np.array([obs.x, obs.y], dtype=np.float32)

                diff = self.pos - obs_pos
                dist = np.linalg.norm(diff)
                
                if hasattr(obs, 'radius') and obs.radius > 0:
                    dist = max(dist - (obs.radius + self.radius), 1e-6)
                
                if dist < 1e-6:
                    angle = np.random.uniform(0, 2 * np.pi)
                    diff = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
                    dist = 1e-6

            # Apply repulsive force within influence zone
            if dist <= rho_0:
                diff_norm = np.linalg.norm(diff)
                if diff_norm > 1e-6:
                    diff_unit = diff / diff_norm
                else:
                    diff_unit = np.array([1.0, 0.0], dtype=np.float32)

                if dist < 0.1:
                    magnitude = eta * 10.0
                else:
                    magnitude = eta * (rho_0 - dist) / (dist * rho_0)
                
                force += magnitude * diff_unit

        return force

    def angle_between(self, v1, v2):
        """Calculate angle between two vectors"""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
            
        cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
        return np.arccos(cos_theta)

    def _create_wall_obstacles(self, bounds):
        """Create virtual wall obstacles"""
        if not bounds:
            return []
            
        env_width, env_height = bounds
        wall_obstacles = []
        wall_density = 0.5
        
        # Create wall points
        num_points_vertical = max(3, int(env_height / wall_density))
        num_points_horizontal = max(3, int(env_width / wall_density))
        
        # Left and right walls
        for i in range(num_points_vertical):
            y = (i / (num_points_vertical - 1)) * env_height
            wall_obstacles.extend([
                type('WallObstacle', (), {'x': 0.0, 'y': float(y), 'radius': 0.0})(),
                type('WallObstacle', (), {'x': float(env_width), 'y': float(y), 'radius': 0.0})()
            ])
        
        # Top and bottom walls
        for i in range(num_points_horizontal):
            x = (i / (num_points_horizontal - 1)) * env_width
            wall_obstacles.extend([
                type('WallObstacle', (), {'x': float(x), 'y': 0.0, 'radius': 0.0})(),
                type('WallObstacle', (), {'x': float(x), 'y': float(env_height), 'radius': 0.0})()
            ])
        
        return wall_obstacles