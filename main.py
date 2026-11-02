"""A small first-person 3D environment rendered with pygame ray casting."""

from __future__ import annotations

import argparse
import math
from typing import Optional

import pygame
from pygame.math import Vector2


SCREEN_WIDTH = 1_000
SCREEN_HEIGHT = 650
HALF_HEIGHT = SCREEN_HEIGHT // 2
FOV = math.radians(66)
NUM_RAYS = 400
MAX_DEPTH = 20.0
MOVE_SPEED = 3.2
ROTATE_SPEED = 2.2
MOUSE_SENSITIVITY = 0.0025

WORLD_MAP = (
    "111111111111111",
    "1.............1",
    "1...111.......1",
    "1...1.........1",
    "1...1..1111...1",
    "1......1......1",
    "1......1..11..1",
    "1.............1",
    "1..111........1",
    "1.............1",
    "1......111....1",
    "1.............1",
    "111111111111111",
)
MAP_WIDTH = len(WORLD_MAP[0])
MAP_HEIGHT = len(WORLD_MAP)

SKY = (31, 48, 76)
FLOOR = (49, 42, 42)
WHITE = (235, 239, 230)
ACCENT = (241, 183, 74)


class Player:
    """Player position, facing direction, and collision-aware movement."""

    def __init__(self) -> None:
        self.position = Vector2(2.5, 2.5)
        self.angle = 0.0
        self.radius = 0.18

    @property
    def direction(self) -> Vector2:
        return Vector2(math.cos(self.angle), math.sin(self.angle))

    def move(self, movement: Vector2, delta_time: float) -> None:
        offset = movement * delta_time
        self._try_move(Vector2(offset.x, 0))
        self._try_move(Vector2(0, offset.y))

    def _try_move(self, offset: Vector2) -> None:
        candidate = self.position + offset
        if not collides(candidate, self.radius):
            self.position = candidate


def is_wall(x: int, y: int) -> bool:
    return (
        x < 0
        or y < 0
        or x >= MAP_WIDTH
        or y >= MAP_HEIGHT
        or WORLD_MAP[y][x] == "1"
    )


def collides(position: Vector2, radius: float) -> bool:
    """Check the four corners of a circular player bounding box."""
    for corner_x in (position.x - radius, position.x + radius):
        for corner_y in (position.y - radius, position.y + radius):
            if is_wall(math.floor(corner_x), math.floor(corner_y)):
                return True
    return False


def cast_ray(origin: Vector2, ray_angle: float) -> tuple[float, int, float]:
    """Return perpendicular distance, wall side, and wall coordinate for a ray."""
    ray_direction = Vector2(math.cos(ray_angle), math.sin(ray_angle))
    map_x, map_y = int(origin.x), int(origin.y)
    delta_distance_x = (
        abs(1 / ray_direction.x) if ray_direction.x else float("inf")
    )
    delta_distance_y = (
        abs(1 / ray_direction.y) if ray_direction.y else float("inf")
    )

    if ray_direction.x < 0:
        step_x, side_distance_x = -1, (origin.x - map_x) * delta_distance_x
    else:
        step_x, side_distance_x = 1, (map_x + 1.0 - origin.x) * delta_distance_x
    if ray_direction.y < 0:
        step_y, side_distance_y = -1, (origin.y - map_y) * delta_distance_y
    else:
        step_y, side_distance_y = 1, (map_y + 1.0 - origin.y) * delta_distance_y

    side = 0
    for _ in range(int(MAX_DEPTH * 2)):
        if side_distance_x < side_distance_y:
            side_distance_x += delta_distance_x
            map_x += step_x
            side = 0
        else:
            side_distance_y += delta_distance_y
            map_y += step_y
            side = 1
        if is_wall(map_x, map_y):
            distance = (
                side_distance_x - delta_distance_x
                if side == 0
                else side_distance_y - delta_distance_y
            )
            hit_coordinate = (
                origin.y + distance * ray_direction.y
                if side == 0
                else origin.x + distance * ray_direction.x
            )
            return max(distance, 0.001), side, hit_coordinate % 1
    return MAX_DEPTH, side, 0.0


def wall_color(distance: float, side: int, texture_coordinate: float) -> tuple[int, int, int]:
    """Create simple brick-like shading without external texture files."""
    brightness = max(0.25, 1.0 - distance / MAX_DEPTH)
    if side == 1:
        brightness *= 0.72
    if int(texture_coordinate * 8) % 2:
        brightness *= 0.86
    base = (191, 116, 63)
    return tuple(int(channel * brightness) for channel in base)


def render_world(surface: pygame.Surface, player: Player) -> None:
    surface.fill(SKY, (0, 0, SCREEN_WIDTH, HALF_HEIGHT))
    surface.fill(FLOOR, (0, HALF_HEIGHT, SCREEN_WIDTH, HALF_HEIGHT))

    ray_width = SCREEN_WIDTH / NUM_RAYS
    for ray_index in range(NUM_RAYS):
        screen_x = (ray_index + 0.5) * ray_width
        ray_angle = player.angle - FOV / 2 + FOV * screen_x / SCREEN_WIDTH
        distance, side, texture_coordinate = cast_ray(player.position, ray_angle)
        corrected_distance = distance * math.cos(ray_angle - player.angle)
        wall_height = min(SCREEN_HEIGHT * 2, SCREEN_HEIGHT / corrected_distance)
        top = int(HALF_HEIGHT - wall_height / 2)
        color = wall_color(corrected_distance, side, texture_coordinate)
        pygame.draw.rect(
            surface,
            color,
            (int(ray_index * ray_width), top, math.ceil(ray_width), int(wall_height)),
        )


def render_minimap(surface: pygame.Surface, player: Player) -> None:
    scale = 12
    origin = Vector2(18, 18)
    panel = pygame.Rect(
        int(origin.x - 6),
        int(origin.y - 6),
        MAP_WIDTH * scale + 12,
        MAP_HEIGHT * scale + 12,
    )
    pygame.draw.rect(surface, (12, 17, 24), panel, border_radius=5)
    for y, row in enumerate(WORLD_MAP):
        for x, tile in enumerate(row):
            color = (106, 68, 51) if tile == "1" else (42, 52, 60)
            pygame.draw.rect(
                surface, color, (int(origin.x + x * scale), int(origin.y + y * scale), scale - 1, scale - 1)
            )
    player_point = origin + player.position * scale
    pygame.draw.circle(surface, ACCENT, player_point, 3)
    direction = player.direction * 9
    pygame.draw.line(surface, ACCENT, player_point, player_point + direction, 2)


def draw_hud(surface: pygame.Surface, font: pygame.font.Font, player: Player) -> None:
    title = font.render("RAYCAST // FIRST PERSON", True, WHITE)
    controls = font.render("WASD move   Arrows / mouse look   ESC quit", True, WHITE)
    position = font.render(
        f"POS {player.position.x:05.2f}, {player.position.y:05.2f}",
        True,
        ACCENT,
    )
    surface.blit(title, (20, SCREEN_HEIGHT - 62))
    surface.blit(controls, (20, SCREEN_HEIGHT - 38))
    surface.blit(position, (SCREEN_WIDTH - position.get_width() - 20, 20))
    pygame.draw.line(surface, WHITE, (SCREEN_WIDTH // 2 - 7, HALF_HEIGHT), (SCREEN_WIDTH // 2 + 7, HALF_HEIGHT), 1)
    pygame.draw.line(surface, WHITE, (SCREEN_WIDTH // 2, HALF_HEIGHT - 7), (SCREEN_WIDTH // 2, HALF_HEIGHT + 7), 1)


def handle_input(player: Player, delta_time: float) -> bool:
    """Process events and apply keyboard/mouse first-person controls."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False
        if event.type == pygame.MOUSEMOTION and pygame.mouse.get_focused():
            player.angle = (player.angle + event.rel[0] * MOUSE_SENSITIVITY) % math.tau

    keys = pygame.key.get_pressed()
    forward = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
    strafe = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
    movement = player.direction * forward
    movement += Vector2(-player.direction.y, player.direction.x) * strafe
    if movement.length_squared() > 1:
        movement.scale_to_length(1)
    player.move(movement * MOVE_SPEED, delta_time)

    rotation = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
    player.angle = (player.angle + rotation * ROTATE_SPEED * delta_time) % math.tau
    return True


def run(max_frames: Optional[int] = None) -> None:
    pygame.init()
    pygame.display.set_caption("RayCast - 3D Environment")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    player = Player()
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    running = True
    frame_count = 0
    while running and (max_frames is None or frame_count < max_frames):
        delta_time = min(clock.tick(60) / 1_000, 0.1)
        running = handle_input(player, delta_time)
        render_world(screen, player)
        render_minimap(screen, player)
        draw_hud(screen, font, player)
        pygame.display.flip()
        frame_count += 1
    pygame.event.set_grab(False)
    pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, help="exit after this many frames (useful for smoke tests)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().frames)