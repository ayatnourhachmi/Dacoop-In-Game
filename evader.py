import pygame
import numpy as np
import os
import random

class Evader:
    def __init__(self, pos=None, scale=50, radius=0.4, de=0.6, image=None):
        self.pos = list(pos) if pos else [0.0, 0.0]
        self.scale = scale
        self.radius = radius  # Visual radius for rendering
        self.de = de          # Collision radius for capture detection
        self.image = None
        self.direction = np.array([0.0, 1.0])  # Movement direction
        
        # Behavior state variables
        self.wall_following = False
        self.zigzag_flag = False
        self.slip_flag = False
        self.zigzag_count = 0
        self.zigzag_last = np.zeros(2)
        self.last_e = np.zeros(2)
        
        # Behavior parameters
        self.sensitivity_range = 3.0  # Distance at which evader starts reacting to agents
        self.zigzag_threshold = 0.3   # Panic level threshold for zigzag behavior
        self.zigzag_min = 15          # Minimum zigzag duration steps
        self.zigzag_max = 30          # Maximum zigzag duration steps
        
        # Corner mode persistence
        self._corner_mode_steps = 0
        self._corner_tangent = np.array([0.0, 1.0])

        # Load sprite image
        if image and os.path.exists(image):
            try:
                loaded_img = pygame.image.load(image)
                try:
                    self.image = loaded_img.convert_alpha()
                except:
                    self.image = loaded_img.convert()
                # Scale image to match evader radius
                diameter = int(self.radius * 2 * self.scale)
                self.image = pygame.transform.scale(self.image, (diameter, diameter))
            except Exception as e:
                print(f"Error loading evader image '{image}': {e}")
                self.image = None

    def set_random_position(self, width_m, height_m, obstacles, clearance=0.6, max_attempts=100):
        """Set evader to a random valid position away from obstacles and walls"""
        for _ in range(max_attempts):
            x = random.uniform(clearance, width_m - clearance)
            y = random.uniform(max(clearance, height_m * 0.7), height_m - clearance)  # Prefer upper area
            if all(not obs.collides_with_point((x, y), self.de) for obs in obstacles):
                self.pos = [x, y]
                return True
        self.pos = [width_m/2, height_m * 0.8]  # Fallback position
        return False

    def reset_state(self):
        """Reset evader behavioral state"""
        self.direction = np.array([0.0, 1.0])
        self.wall_following = False
        self.zigzag_flag = False
        self.slip_flag = False
        self.zigzag_count = 0
        self.zigzag_last = np.zeros(2)
        self.last_e = np.zeros(2)
        self._corner_mode_steps = 0

    def compute_escape_direction(self, agents, obstacles, world_bounds, dt):
        """
        Compute desired evader movement direction using zigzag/escape logic
        
        Args:
            agents: List of agent objects
            obstacles: List of obstacle objects
            world_bounds: (width, height) of the world
            dt: Time step
            
        Returns:
            np.array: Unit direction vector for movement
        """
        world_width, world_height = world_bounds
        clearance = max(self.de * 2.0, 0.6)
        
        # Decrease corner mode persistence counter
        if self._corner_mode_steps > 0:
            self._corner_mode_steps -= 1
        
        # Corner detection
        near_left = self.pos[0] <= clearance + 1e-6
        near_right = self.pos[0] >= world_width - clearance - 1e-6
        near_top = self.pos[1] <= clearance + 1e-6
        near_bottom = self.pos[1] >= world_height - clearance - 1e-6
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
                wall_normal = -self.direction
            wall_normal = wall_normal / (np.linalg.norm(wall_normal) + 1e-9)
            t1 = np.array([-wall_normal[1], wall_normal[0]])
            t2 = -t1
            # Prefer tangent aligned with current direction to avoid jitter
            cand = t1 if np.dot(t1, self.direction) >= np.dot(t2, self.direction) else t2
            # Persist choice
            self._corner_tangent = cand
            self._corner_mode_steps = max(self._corner_mode_steps, 12)
            self.wall_following = True
            return cand / (np.linalg.norm(cand) + 1e-9)

        # Calculate distances to active agents
        agent_positions = [a.pos for a in agents if not a.reached_evader]
        if len(agent_positions) == 0:
            mean_escape = np.zeros(2)
            nearest_agent_dist = float('inf')
        else:
            diffs = [self.pos - np.array(ap) for ap in agent_positions]
            dists = [np.linalg.norm(d) for d in diffs]
            nearest_agent_dist = min(dists)
            # Weighted average of away vectors (inverse-square like)
            contribs = []
            for d, r in zip(diffs, dists):
                if r > 1e-6:
                    contribs.append(d / (r * r))
            mean_escape = np.mean(contribs, axis=0) if contribs else np.zeros(2)

        # Calculate panic level based on nearest agent distance
        if nearest_agent_dist == float('inf') or nearest_agent_dist > self.sensitivity_range:
            p_panic = 0.0
        else:
            x = -nearest_agent_dist / self.sensitivity_range + 1.0
            p_panic = (np.exp(x) - 1.0) / (np.e - 1.0)

        # Compute minimum distance to obstacles
        def dist_to_obstacle(obs):
            if hasattr(obs, 'width_m') and hasattr(obs, 'height_m'):
                left = obs.x - obs.width_m / 2
                right = obs.x + obs.width_m / 2
                top = obs.y - obs.height_m / 2
                bottom = obs.y + obs.height_m / 2
                cx = np.clip(self.pos[0], left, right)
                cy = np.clip(self.pos[1], top, bottom)
                return np.linalg.norm(self.pos - np.array([cx, cy]))
            else:
                obs_pos = np.array([obs.x, obs.y])
                return np.linalg.norm(self.pos - obs_pos)

        min_obs_dist = min([dist_to_obstacle(o) for o in obstacles], default=float('inf'))

        # Zigzag behavior when not panicked and no close obstacle
        if p_panic < self.zigzag_threshold and min_obs_dist > 0.3:
            self.zigzag_flag = True
            if self.zigzag_count == 0:
                # Pick a random direction roughly forward
                while True:
                    ang = np.random.uniform(-np.pi, np.pi)
                    cand = np.array([np.cos(ang), np.sin(ang)])
                    if np.dot(cand, self.direction) > 0:
                        self.zigzag_last = cand
                        break
                self.zigzag_count = 1
                F_escape = self.zigzag_last
            else:
                F_escape = self.zigzag_last
                self.zigzag_count += 1
                if self.zigzag_count > np.random.randint(self.zigzag_min, self.zigzag_max + 1):
                    self.zigzag_count = 0
        else:
            self.zigzag_flag = False
            self.zigzag_count = 0
            # Use averaged away vector
            if np.linalg.norm(mean_escape) > 1e-6:
                F_escape = mean_escape / np.linalg.norm(mean_escape)
            else:
                F_escape = self.direction.copy()

        # Predictive collision check
        step_preview = 1.0 * dt  # Preview distance
        cand_pos = np.array(self.pos) + F_escape * step_preview

        def would_hit_walls(p):
            return (
                p[0] < clearance or p[0] > world_width - clearance or
                p[1] < clearance or p[1] > world_height - clearance
            )

        def would_hit_obstacles(p):
            return any(obs.collides_with_point(p, self.de) for obs in obstacles)

        self.wall_following = False
        if would_hit_walls(cand_pos) or would_hit_obstacles(cand_pos):
            # Choose tangential direction around nearest constraint
            # Prefer nearest obstacle if any
            nearest_o = None
            if len(obstacles) > 0:
                dists = [(dist_to_obstacle(o), o) for o in obstacles]
                dists.sort(key=lambda x: x[0])
                nearest_o = dists[0][1]
            
            if nearest_o is not None and dist_to_obstacle(nearest_o) < float('inf'):
                if hasattr(nearest_o, 'width_m') and hasattr(nearest_o, 'height_m'):
                    left = nearest_o.x - nearest_o.width_m / 2
                    right = nearest_o.x + nearest_o.width_m / 2
                    top = nearest_o.y - nearest_o.height_m / 2
                    bottom = nearest_o.y + nearest_o.height_m / 2
                    cx = np.clip(self.pos[0], left, right)
                    cy = np.clip(self.pos[1], top, bottom)
                    closest = np.array([cx, cy])
                else:
                    closest = np.array([nearest_o.x, nearest_o.y])
                normal = self.pos - closest
            else:
                # Use wall normal
                normal = np.array([0.0, 0.0])
                if self.pos[0] < clearance:
                    normal += np.array([1.0, 0.0])
                if self.pos[0] > world_width - clearance:
                    normal += np.array([-1.0, 0.0])
                if self.pos[1] < clearance:
                    normal += np.array([0.0, 1.0])
                if self.pos[1] > world_height - clearance:
                    normal += np.array([0.0, -1.0])
                if np.linalg.norm(normal) < 1e-6:
                    normal = -F_escape
            
            if np.linalg.norm(normal) > 1e-6:
                normal = normal / np.linalg.norm(normal)
                tangential = np.array([-normal[1], normal[0]])
                if np.dot(tangential, self.direction) < 0:
                    tangential = -tangential
                F_total = tangential
                self.wall_following = True
            else:
                F_total = F_escape
        else:
            F_total = F_escape

        # Normalize final direction
        if np.linalg.norm(F_total) > 1e-6:
            F_total = F_total / np.linalg.norm(F_total)
        else:
            F_total = self.direction.copy()
        
        return F_total

    def update_manual_control(self, speed, dt, world_bounds, obstacles):
        """Update evader position via manual keyboard control (arrow keys/WASD)"""
        try:
            import pygame
            keys = pygame.key.get_pressed()
            dx = (1 if keys[pygame.K_RIGHT] or keys[pygame.K_d] else 0) - (1 if keys[pygame.K_LEFT] or keys[pygame.K_a] else 0)
            dy = (1 if keys[pygame.K_DOWN] or keys[pygame.K_s] else 0) - (1 if keys[pygame.K_UP] or keys[pygame.K_w] else 0)
        except Exception:
            dx, dy = 0, 0

        move = np.array([dx, dy], dtype=float)
        if np.linalg.norm(move) > 1e-6:
            move_dir = move / np.linalg.norm(move)
            self.direction = move_dir
            step = speed * dt
            candidate = np.array(self.pos) + move_dir * step
            
            # Keep wall clearance
            world_width, world_height = world_bounds
            clearance = max(self.de * 2.0, 0.6)
            candidate[0] = np.clip(candidate[0], clearance, world_width - clearance)
            candidate[1] = np.clip(candidate[1], clearance, world_height - clearance)
            
            # Skip move if collides with obstacles
            if not any(obs.collides_with_point(candidate, self.de) for obs in obstacles):
                self.pos[:] = candidate

    def update_autonomous(self, agents, obstacles, world_bounds, speed, dt):
        """Update evader position using autonomous behavior"""
        desired_direction = self.compute_escape_direction(agents, obstacles, world_bounds, dt)
        self.direction = desired_direction
        
        step = speed * dt
        candidate = np.array(self.pos) + desired_direction * step
        
        # Keep wall clearance
        world_width, world_height = world_bounds
        clearance = max(self.de * 2.0, 0.6)
        candidate[0] = np.clip(candidate[0], clearance, world_width - clearance)
        candidate[1] = np.clip(candidate[1], clearance, world_height - clearance)
        
        # Apply movement if no collision
        if not any(obs.collides_with_point(candidate, self.de) for obs in obstacles):
            self.pos[:] = candidate

    def draw(self, screen, scale):
        """Draw the evader on the screen"""
        px, py = int(self.pos[0] * scale), int(self.pos[1] * scale)
        
        # Draw evader collision radius (de) as a red circle outline
        pygame.draw.circle(screen, (255, 0, 0), (px, py), int(self.de * scale), 2)
        
        # Draw evader visual radius as a yellow circle outline
        pygame.draw.circle(screen, (255, 255, 0), (px, py), int(self.radius * scale), 1)
        
        if self.image:
            # Rotate sprite to face movement direction (assumes artwork points up)
            if np.linalg.norm(self.direction) > 1e-6:
                angle_deg = -np.degrees(np.arctan2(self.direction[1], self.direction[0])) - 90
            else:
                angle_deg = 0.0
            rotated = pygame.transform.rotate(self.image, angle_deg)
            rect = rotated.get_rect(center=(px, py))
            screen.blit(rotated, rect)
        else:
            # Draw filled circle for the evader center
            pygame.draw.circle(screen, (255, 215, 0), (px, py), int(self.radius * scale * 0.7))

    def get_status_info(self):
        """Get current status information for debugging"""
        status = "NORMAL"
        if self.wall_following:
            status = "WALL_FOLLOWING"
        if self.zigzag_flag:
            status = "ZIGZAG"
        if self.slip_flag:
            status = "SLIP"
        
        return {
            'position': self.pos.copy(),
            'direction': self.direction.copy(),
            'status': status,
            'wall_following': self.wall_following,
            'zigzag_flag': self.zigzag_flag,
            'slip_flag': self.slip_flag,
            'visual_radius': self.radius,
            'collision_radius': self.de
        }