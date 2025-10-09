import pygame
import numpy as np
import os
import random

class Obstacle:
    def __init__(self, x, y, width_m, height_m, scale, color=(255, 255, 255), image=None, show_bounds=False, image_scale_mode="bounds"):
        self.x = x
        self.y = y
        self.width_m = width_m
        self.height_m = height_m
        self.scale = scale
        self.color = color
        self.image = None
        self.show_bounds = show_bounds
        self.scaled_image_size = None  # Track actual image size after scaling
        self.image_scale_mode = image_scale_mode  # "bounds", "fill", or "fit"

        if image and os.path.exists(image):
            try:
                loaded_img = pygame.image.load(image)
                # Get original image dimensions for reference
                self.original_size = loaded_img.get_size()
                
                # Calculate target size based on obstacle dimensions
                target_width = int(self.width_m * self.scale)
                target_height = int(self.height_m * self.scale)
                
                if self.image_scale_mode == "fill":
                    # Fill the entire bounds (may stretch image)
                    new_width = target_width
                    new_height = target_height
                    print(f"Obstacle image '{image}': FILL mode - stretching to fill bounds")
                    
                elif self.image_scale_mode == "bounds":
                    # Scale to fit bounds while preserving aspect ratio (default behavior)
                    image_aspect = self.original_size[0] / self.original_size[1]
                    obstacle_aspect = self.width_m / self.height_m
                    
                    if image_aspect > obstacle_aspect:
                        new_width = target_width
                        new_height = int(target_width / image_aspect)
                    else:
                        new_height = target_height
                        new_width = int(target_height * image_aspect)
                    print(f"Obstacle image '{image}': BOUNDS mode - fit within bounds")
                    
                else:  # "fit" mode - scale image to a reasonable size regardless of bounds
                    # Scale image to a fixed proportion of the obstacle size for PNG transparency handling
                    scale_factor = 0.8  # Use 80% of the obstacle bounds
                    new_width = int(target_width * scale_factor)
                    new_height = int(target_height * scale_factor)
                    print(f"Obstacle image '{image}': FIT mode - scaled to {scale_factor*100}% of bounds")
                
                # Store the actual scaled size
                self.scaled_image_size = (new_width, new_height)
                
                print(f"  Original size: {self.original_size}")
                print(f"  Target bounds: {target_width}x{target_height}")
                print(f"  Scaled to: {new_width}x{new_height} (preserving aspect ratio)")
                
                try:
                    self.image = loaded_img.convert_alpha()
                except:
                    self.image = loaded_img.convert()
                
                # Scale with preserved aspect ratio
                self.image = pygame.transform.scale(self.image, (new_width, new_height))
                
            except Exception as e:
                print(f"Error loading obstacle image '{image}': {e}")
                self.image = None
        else:
            if image:
                print(f"Obstacle image '{image}' not found, using colored rectangle")

    def draw(self, surface):
        # Draw the collision bounds first (if enabled) so it appears behind the image
        if self.show_bounds:
            bounds_rect = self.get_rect()
            pygame.draw.rect(surface, (255, 0, 0), bounds_rect, 2)  # Red border
            # Draw center point
            center_x, center_y = int(self.x * self.scale), int(self.y * self.scale)
            pygame.draw.circle(surface, (255, 0, 0), (center_x, center_y), 3)
        
        # Draw the image or colored rectangle
        if self.image:
            # Center the image within the obstacle bounds
            rect = self.image.get_rect(center=(int(self.x*self.scale), int(self.y*self.scale)))
            surface.blit(self.image, rect)
        else:
            pygame.draw.rect(surface, self.color, self.get_rect())
        
        # Draw bounds info text on top
        if self.show_bounds:
            bounds_rect = self.get_rect()
            try:
                pygame.font.init()
                font = pygame.font.Font(None, 20)
                # Show both collision bounds and image size
                text1 = font.render(f"Bounds: {self.width_m}x{self.height_m}m", True, (255, 255, 255))
                surface.blit(text1, (bounds_rect.x, bounds_rect.y - 40))
                
                if self.scaled_image_size:
                    img_w, img_h = self.scaled_image_size
                    text2 = font.render(f"Image: {img_w}x{img_h}px", True, (200, 200, 200))
                    surface.blit(text2, (bounds_rect.x, bounds_rect.y - 20))
            except:
                pass

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

        # Draw evader (now using the evader object's own draw method)
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

        pygame.display.flip()

    def tick(self, fps):
        self.clock.tick(fps)

    def close(self):
        pygame.quit()