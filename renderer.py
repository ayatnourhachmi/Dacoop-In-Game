import pygame
import numpy as np
import os
import random

class Obstacle:
    def __init__(self, x, y, width_m, height_m, scale, color=(255, 255, 255), image=None):
        self.x = x
        self.y = y
        self.width_m = width_m
        self.height_m = height_m
        self.scale = scale
        self.color = color
        self.image = None

        if image and os.path.exists(image):
            try:
                loaded_img = pygame.image.load(image)
                try:
                    self.image = loaded_img.convert_alpha()
                except:
                    self.image = loaded_img.convert()
                self.image = pygame.transform.scale(
                    self.image,
                    (int(self.width_m*self.scale), int(self.height_m*self.scale))
                )
            except Exception as e:
                print(f"Error loading obstacle image '{image}': {e}")
                self.image = None

    def draw(self, surface):
        if self.image:
            rect = self.image.get_rect(center=(int(self.x*self.scale), int(self.y*self.scale)))
            surface.blit(self.image, rect)
        else:
            pygame.draw.rect(surface, self.color, self.get_rect())

    def get_rect(self):
        return pygame.Rect(
            int((self.x - self.width_m/2)*self.scale),
            int((self.y - self.height_m/2)*self.scale),
            int(self.width_m*self.scale),
            int(self.height_m*self.scale)
        )

    def distance_to_point(self, point):
        px, py = point
        left, right = self.x - self.width_m/2, self.x + self.width_m/2
        top, bottom = self.y - self.height_m/2, self.y + self.height_m/2
        dx = max(left - px, 0, px - right)
        dy = max(top - py, 0, py - bottom)
        return np.sqrt(dx*dx + dy*dy)

    def collides_with_point(self, point, collision_radius):
        return self.distance_to_point(point) < collision_radius

pygame.init()
pygame.display.set_mode((1, 1))  # Minimal display just to allow convert/convert_alpha

class Evader:
    def __init__(self, pos=None, scale=50, radius=0.4, de=0.6, image=None):
        self.pos = list(pos) if pos else [0.0, 0.0]
        self.scale = scale
        self.radius = radius  # Visual radius for rendering
        self.de = de          # Collision radius for capture detection
        self.image = None

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

    def set_random_position(self, width_m, height_m, obstacles, max_attempts=100):
        for _ in range(max_attempts):
            x = random.uniform(self.de, width_m - self.de)  # Use collision radius for positioning
            y = random.uniform(self.de, height_m - self.de)
            if all(not obs.collides_with_point((x, y), self.de) for obs in obstacles):
                self.pos = [x, y]
                return True
        self.pos = [width_m/2, height_m/2]
        return False

    def draw(self, screen, scale):
        px, py = int(self.pos[0] * scale), int(self.pos[1] * scale)
        
        # Draw evader collision radius (de) as a red circle outline
        pygame.draw.circle(screen, (255, 0, 0), (px, py), int(self.de * scale), 2)
        
        # Draw evader visual radius as a yellow circle outline
        pygame.draw.circle(screen, (255, 255, 0), (px, py), int(self.radius * scale), 1)
        
        if self.image:
            # Rotate sprite to face movement direction (assumes artwork points up)
            direction = getattr(self, 'direction', None)
            if direction is None:
                # fallback: infer from last position difference is not stored; use no rotation
                angle_deg = 0.0
            else:
                if np.linalg.norm(direction) > 1e-6:
                    angle_deg = -np.degrees(np.arctan2(direction[1], direction[0])) - 90
                else:
                    angle_deg = 0.0
            rotated = pygame.transform.rotate(self.image, angle_deg)
            rect = rotated.get_rect(center=(px, py))
            screen.blit(rotated, rect)
        else:
            # Draw filled circle for the evader center
            pygame.draw.circle(screen, (255, 215, 0), (px, py), int(self.radius * scale * 0.7))

class Renderer:
    def __init__(self, width_m, height_m, scale, agent_radius,
                 obstacles=None, agent_image=None, background_image=None):
        pygame.init()
        self.width_m = width_m
        self.height_m = height_m
        self.scale = scale
        self.window_size = (int(width_m*scale), int(height_m*scale))
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Continuous World Env - Evader Chase")
        self.clock = pygame.time.Clock()

        self.agent_radius = agent_radius

        # Obstacles
        self.obstacles = obstacles if obstacles else []

        # Background
        self.background = None
        if background_image and os.path.exists(background_image):
            try:
                bg = pygame.image.load(background_image)
                try:
                    self.background = bg.convert()
                except:
                    self.background = bg
                self.background = pygame.transform.scale(self.background, self.window_size)
            except:
                self.background = None

        # Agent sprite
        self.agent_sprite = None
        if agent_image and os.path.exists(agent_image):
            try:
                agent_img = pygame.image.load(agent_image)
                try:
                    self.agent_sprite = agent_img.convert_alpha()
                except:
                    self.agent_sprite = agent_img.convert()
                size = (int(agent_radius*4*scale), int(agent_radius*4*scale))
                self.agent_sprite = pygame.transform.scale(self.agent_sprite, size)
            except:
                self.agent_sprite = None

    def draw(self, agent_positions, evader=None, agents=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Draw background
        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill((50, 50, 50))

        # Draw obstacles
        for obs in self.obstacles:
            obs.draw(self.screen)

        # Draw evader
        if evader:
            evader.draw(self.screen, self.scale)

        # Draw agents
        colors = [(255, 215, 0), (0, 191, 255), (255, 0, 255), (0, 255, 0), (255, 165, 0)]
        for i, pos in enumerate(agent_positions):
            px, py = int(pos[0] * self.scale), int(pos[1] * self.scale)

            # Agent sprite or circle
            if self.agent_sprite:
                if agents is not None and hasattr(agents[i], 'orientation'):
                    ori = agents[i].orientation
                    angle_deg = -np.degrees(np.arctan2(ori[1], ori[0]))
                    angle_deg -= 90  # artwork points up: apply -90° offset
                    rotated = pygame.transform.rotate(self.agent_sprite, angle_deg)
                    rect = rotated.get_rect(center=(px, py))
                    self.screen.blit(rotated, rect)
                else:
                    rect = self.agent_sprite.get_rect(center=(px, py))
                    self.screen.blit(self.agent_sprite, rect)
            else:
                color = colors[i % len(colors)]
                pygame.draw.circle(self.screen, color, (px, py), int(self.agent_radius * self.scale))

            # --- Debug circles: dc (red), ds (blue) ---
            if agents is not None:
                dc = agents[i].dc
                ds = agents[i].ds
                pygame.draw.circle(self.screen, (255, 0, 0), (px, py), int(dc * self.scale), 1)
                pygame.draw.circle(self.screen, (0, 0, 255), (px, py), int(ds * self.scale), 1)

                # (Heading line removed as per request)

        pygame.display.flip()

    def tick(self, fps):
        self.clock.tick(fps)

    def close(self):
        pygame.quit()