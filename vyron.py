"""Developer visualizer for the ray-casting demo.

Run with:
    python developer_view.py

This file intentionally imports the game logic from ``main.py`` without
modifying it. It adds a top-down debugging view, ray traces, DDA traversal,
projection measurements, and live player/ray statistics.
"""

from __future__ import annotations

import argparse
import math
from typing import Optional

import cv2
import numpy as np
import pygame
from pygame.math import Vector2

import main
from vyron_brain import BrainAction, Movement, VyronBrain


WINDOW_WIDTH = 1_360
WINDOW_HEIGHT = 820
WORLD_PANEL = pygame.Rect(18, 18, 700, 590)
INFO_PANEL = pygame.Rect(740, 18, 602, 784)
MAP_SCALE = 40
MAP_ORIGIN = Vector2(32, 54)
DEBUG_RAY_COUNT = 25
TRACE_RAY_COUNT = 5

BACKGROUND = (13, 18, 25)
PANEL = (22, 29, 39)
GRID = (40, 51, 65)
WALL = (135, 80, 52)
OPEN = (27, 38, 48)
TEXT = (221, 229, 235)
MUTED = (148, 163, 177)
RAY = (246, 193, 81)
TRACE = (255, 104, 100)
PLAYER = (96, 220, 147)
HIT = (255, 142, 70)
PLAYER_VIEW_WIDTH = 900
PLAYER_VIEW_HEIGHT = 580
PLAYER_VIEW_NAME = "VYRON // LIVE PLAYER VIEW"
AUTONOMOUS_ACTION_INTERVAL = 0.1


def world_to_screen(position: Vector2) -> Vector2:
    return MAP_ORIGIN + position * MAP_SCALE


def dda_trace(origin: Vector2, angle: float) -> tuple[list[Vector2], Vector2, int, float]:
    """Return every grid boundary visited by DDA and the final wall hit."""
    direction = Vector2(math.cos(angle), math.sin(angle))
    map_x, map_y = int(origin.x), int(origin.y)
    delta_x = abs(1 / direction.x) if direction.x else float("inf")
    delta_y = abs(1 / direction.y) if direction.y else float("inf")

    if direction.x < 0:
        step_x, side_x = -1, (origin.x - map_x) * delta_x
    else:
        step_x, side_x = 1, (map_x + 1 - origin.x) * delta_x
    if direction.y < 0:
        step_y, side_y = -1, (origin.y - map_y) * delta_y
    else:
        step_y, side_y = 1, (map_y + 1 - origin.y) * delta_y

    points = [origin.copy()]
    side = 0
    distance = main.MAX_DEPTH
    for _ in range(int(main.MAX_DEPTH * 2)):
        if side_x < side_y:
            side_x += delta_x
            map_x += step_x
            side = 0
            distance = side_x - delta_x
        else:
            side_y += delta_y
            map_y += step_y
            side = 1
            distance = side_y - delta_y
        point = origin + direction * distance
        points.append(point)
        if main.is_wall(map_x, map_y):
            return points, point, side, distance
    return points, origin + direction * main.MAX_DEPTH, side, distance


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = TEXT,
) -> None:
    surface.blit(font.render(text, True, color), position)


def draw_world_grid(surface: pygame.Surface, font: pygame.font.Font) -> None:
    pygame.draw.rect(surface, PANEL, WORLD_PANEL, border_radius=8)
    draw_text(surface, font, "WORLD SPACE / DDA RAY DEBUGGER", (32, 28), RAY)
    for y, row in enumerate(main.WORLD_MAP):
        for x, tile in enumerate(row):
            rect = pygame.Rect(
                int(MAP_ORIGIN.x + x * MAP_SCALE),
                int(MAP_ORIGIN.y + y * MAP_SCALE),
                MAP_SCALE - 1,
                MAP_SCALE - 1,
            )
            pygame.draw.rect(surface, WALL if tile == "1" else OPEN, rect)
            pygame.draw.rect(surface, GRID, rect, 1)
            draw_text(surface, font, f"{x},{y}", (rect.x + 3, rect.y + 3), MUTED)


def draw_rays(surface: pygame.Surface, player: main.Player) -> dict[str, float | int]:
    origin = world_to_screen(player.position)
    for ray_index in range(DEBUG_RAY_COUNT):
        screen_ratio = (ray_index + 0.5) / DEBUG_RAY_COUNT
        angle = player.angle - main.FOV / 2 + main.FOV * screen_ratio
        distance, side, coordinate = main.cast_ray(player.position, angle)
        hit = world_to_screen(player.position + Vector2(math.cos(angle), math.sin(angle)) * distance)
        color = RAY if ray_index in (0, DEBUG_RAY_COUNT // 2, DEBUG_RAY_COUNT - 1) else (122, 102, 59)
        pygame.draw.line(surface, color, origin, hit, 2 if color == RAY else 1)
        pygame.draw.circle(surface, HIT if color == RAY else (126, 91, 61), hit, 3)
        if ray_index == DEBUG_RAY_COUNT // 2:
            center_distance, center_side, center_coordinate = distance, side, coordinate

    pygame.draw.circle(surface, PLAYER, origin, 9)
    pygame.draw.line(
        surface,
        PLAYER,
        origin,
        origin + Vector2(math.cos(player.angle), math.sin(player.angle)) * 28,
        3,
    )
    return {
        "distance": center_distance,
        "side": center_side,
        "coordinate": center_coordinate,
    }


def draw_trace(surface: pygame.Surface, player: main.Player) -> int:
    origin = world_to_screen(player.position)
    for index in range(TRACE_RAY_COUNT):
        ratio = (index + 0.5) / TRACE_RAY_COUNT
        angle = player.angle - main.FOV / 2 + main.FOV * ratio
        points, hit, _, _ = dda_trace(player.position, angle)
        scaled_points = [world_to_screen(point) for point in points]
        pygame.draw.lines(surface, TRACE, False, scaled_points, 2)
        for step, point in enumerate(scaled_points[1:], 1):
            pygame.draw.circle(surface, (255, 181, 110), point, 4)
            if index == TRACE_RAY_COUNT // 2:
                pygame.draw.circle(surface, TEXT, point, 7, 1)
                draw_text(surface, pygame.font.Font(None, 16), str(step), (point.x + 5, point.y - 5), TEXT)
        pygame.draw.circle(surface, HIT, world_to_screen(hit), 5)
    pygame.draw.circle(surface, PLAYER, origin, 10, 2)
    return len(dda_trace(player.position, player.angle)[0]) - 1


def draw_projection(surface: pygame.Surface, font: pygame.font.Font, player: main.Player, info: dict[str, float | int]) -> None:
    top = 645
    draw_text(surface, font, "PROJECTION PIPELINE", (32, top), RAY)
    distance = float(info["distance"])
    corrected = distance
    wall_height = min(main.SCREEN_HEIGHT * 2, main.SCREEN_HEIGHT / max(corrected, 0.001))
    values = (
        f"FOV: {math.degrees(main.FOV):.1f} degrees",
        f"center ray angle: {math.degrees(player.angle):.1f} degrees",
        f"perpendicular distance: {distance:.3f} world units",
        f"wall height: {wall_height:.1f} screen pixels",
        f"rays cast by game renderer: {main.NUM_RAYS}",
    )
    for index, value in enumerate(values):
        draw_text(surface, font, value, (32, top + 25 + index * 21), TEXT)


def draw_info(
    surface: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    player: main.Player,
    info: dict[str, float | int],
    trace_steps: int,
    fps: float,
) -> None:
    pygame.draw.rect(surface, PANEL, INFO_PANEL, border_radius=8)
    draw_text(surface, font, "DEVELOPER MODE", (760, 36), RAY)
    draw_text(surface, small_font, "Everything below is live from the current frame.", (760, 66), MUTED)

    sections = [
        ("PLAYER STATE", [
            f"position      ({player.position.x:7.3f}, {player.position.y:7.3f})",
            f"angle         {math.degrees(player.angle):7.2f} degrees",
            f"direction     ({player.direction.x:7.3f}, {player.direction.y:7.3f})",
            f"collision box radius {player.radius:.2f}",
        ]),
        ("RAY DISTRIBUTION", [
            f"field of view {math.degrees(main.FOV):7.2f} degrees",
            f"debug rays    {DEBUG_RAY_COUNT:7d}",
            f"ray spacing   {math.degrees(main.FOV / DEBUG_RAY_COUNT):7.2f} degrees",
            f"game rays     {main.NUM_RAYS:7d}",
            "yellow = edge/center sample rays",
        ]),
        ("CENTER RAY HIT", [
            f"wall distance {float(info['distance']):7.3f} units",
            f"wall side     {'Y boundary' if info['side'] else 'X boundary'}",
            f"hit coordinate {float(info['coordinate']):6.3f} across wall",
            f"DDA steps     {trace_steps:7d}",
        ]),
        ("CONTROLS", [
            "W / S     forward / backward",
            "A / D     strafe left / right",
            "LEFT/RIGHT rotate without mouse",
            "mouse     rotate camera",
            "ESC       quit",
            "F1        toggle this panel",
        ]),
        ("PERFORMANCE", [
            f"FPS         {fps:7.1f}",
            f"map size    {main.MAP_WIDTH} x {main.MAP_HEIGHT} cells",
            f"max depth   {main.MAX_DEPTH:.1f} units",
        ]),
    ]
    y = 112
    for title, lines in sections:
        draw_text(surface, font, title, (760, y), RAY)
        y += 27
        for line in lines:
            draw_text(surface, small_font, line, (760, y), TEXT)
            y += 21
        y += 18


def render_player_view(player: main.Player) -> np.ndarray:
    """Build the live first-person view as an OpenCV BGR frame."""
    frame = np.zeros((PLAYER_VIEW_HEIGHT, PLAYER_VIEW_WIDTH, 3), dtype=np.uint8)
    horizon = PLAYER_VIEW_HEIGHT // 2
    frame[:horizon] = (76, 48, 31)
    frame[horizon:] = (42, 42, 49)

    ray_width = PLAYER_VIEW_WIDTH / main.NUM_RAYS
    for ray_index in range(main.NUM_RAYS):
        screen_x = (ray_index + 0.5) * ray_width
        ray_angle = player.angle - main.FOV / 2 + main.FOV * screen_x / PLAYER_VIEW_WIDTH
        distance, side, texture_coordinate = main.cast_ray(player.position, ray_angle)
        corrected_distance = distance * math.cos(ray_angle - player.angle)
        wall_height = min(PLAYER_VIEW_HEIGHT * 2, PLAYER_VIEW_HEIGHT / max(corrected_distance, 0.001))
        top = max(0, int(horizon - wall_height / 2))
        bottom = min(PLAYER_VIEW_HEIGHT, int(horizon + wall_height / 2))
        color = main.wall_color(corrected_distance, side, texture_coordinate)
        bgr = (color[2], color[1], color[0])
        cv2.rectangle(
            frame,
            (int(ray_index * ray_width), top),
            (int((ray_index + 1) * ray_width), bottom),
            bgr,
            -1,
        )

    cv2.line(frame, (PLAYER_VIEW_WIDTH // 2 - 9, horizon), (PLAYER_VIEW_WIDTH // 2 + 9, horizon), (235, 239, 230), 1)
    cv2.line(frame, (PLAYER_VIEW_WIDTH // 2, horizon - 9), (PLAYER_VIEW_WIDTH // 2, horizon + 9), (235, 239, 230), 1)
    cv2.putText(
        frame,
        "LIVE PLAYER VIEW // RAYCAST OUTPUT",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 239, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"pos ({player.position.x:.2f}, {player.position.y:.2f})  angle {math.degrees(player.angle):.1f} deg",
        (18, PLAYER_VIEW_HEIGHT - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (81, 183, 241),
        1,
        cv2.LINE_AA,
    )
    return frame


def apply_brain_action(player: main.Player, action: BrainAction, delta_time: float) -> None:
    """Apply one validated brain action using the same movement rules as input."""
    movement = Vector2()
    if action.movement == Movement.FORWARD:
        movement += player.direction
    elif action.movement == Movement.BACKWARD:
        movement -= player.direction
    elif action.movement == Movement.LEFT:
        movement += Vector2(player.direction.y, -player.direction.x)
    elif action.movement == Movement.RIGHT:
        movement += Vector2(-player.direction.y, player.direction.x)

    if movement.length_squared() > 1:
        movement.scale_to_length(1)
    player.move(movement * main.MOVE_SPEED, delta_time)
    player.angle = (player.angle + action.rotation * main.ROTATE_SPEED * delta_time) % math.tau


def run(
    max_frames: Optional[int] = None,
    show_opencv: bool = True,
    autonomous: bool = False,
) -> None:
    pygame.init()
    pygame.display.set_caption("RayCast - Developer Visualizer")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    small_font = pygame.font.Font(None, 19)
    player = main.Player()
    brain = VyronBrain()
    next_brain_update = 0.0
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    if show_opencv:
        cv2.namedWindow(PLAYER_VIEW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PLAYER_VIEW_NAME, PLAYER_VIEW_WIDTH, PLAYER_VIEW_HEIGHT)

    show_info = True
    running = True
    frame_count = 0
    while running and (max_frames is None or frame_count < max_frames):
        delta_time = min(clock.tick(60) / 1_000, 0.1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    show_info = not show_info
            elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_focused():
                player.angle = (player.angle + event.rel[0] * main.MOUSE_SENSITIVITY) % math.tau

        if autonomous:
            next_brain_update -= delta_time
            if next_brain_update <= 0:
                action = brain.observe(render_player_view(player))
                next_brain_update = AUTONOMOUS_ACTION_INTERVAL
            else:
                action = brain.last_action
            apply_brain_action(player, action, delta_time)
        else:
            keys = pygame.key.get_pressed()
            movement = player.direction * (float(keys[pygame.K_w]) - float(keys[pygame.K_s]))
            movement += Vector2(-player.direction.y, player.direction.x) * (
                float(keys[pygame.K_d]) - float(keys[pygame.K_a])
            )
            if movement.length_squared() > 1:
                movement.scale_to_length(1)
            player.move(movement * main.MOVE_SPEED, delta_time)
            player.angle = (
                player.angle
                + (float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT]))
                * main.ROTATE_SPEED
                * delta_time
            ) % math.tau

        screen.fill(BACKGROUND)
        draw_world_grid(screen, small_font)
        info = draw_rays(screen, player)
        trace_steps = draw_trace(screen, player)
        draw_projection(screen, small_font, player, info)
        if show_info:
            draw_info(screen, font, small_font, player, info, trace_steps, clock.get_fps())
        else:
            draw_text(screen, font, "F1: show developer information", (760, 40), RAY)
        pygame.display.flip()
        if show_opencv:
            cv2.imshow(PLAYER_VIEW_NAME, render_player_view(player))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False
        frame_count += 1

    pygame.event.set_grab(False)
    if show_opencv:
        cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, help="exit after this many frames")
    parser.add_argument("--no-opencv", action="store_true", help="disable the live OpenCV player window")
    parser.add_argument("--autonomous", action="store_true", help="run through the autonomous brain interface")
    arguments = parser.parse_args()
    run(arguments.frames, show_opencv=not arguments.no_opencv, autonomous=arguments.autonomous)
