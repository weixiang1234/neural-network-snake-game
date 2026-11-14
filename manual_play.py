"""Minimal viewer. A/D or left/right arrows select a relative turn."""

import argparse
import math
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from snake_env import SnakeEnv


def draw(screen, font, env, paused):
    screen.fill((20, 24, 30))
    cell = 24
    origin = (60, 50)
    board = pygame.Rect(*origin, 480, 480)
    pygame.draw.rect(screen, (30, 36, 44), board)
    for i in range(21):
        pygame.draw.line(screen, (38, 44, 52), (60 + i * cell, 50), (60 + i * cell, 530))
        pygame.draw.line(screen, (38, 44, 52), (60, 50 + i * cell), (540, 50 + i * cell))

    def block(position, color):
        x, y = position
        pygame.draw.rect(screen, color, (origin[0] + x * cell + 1, origin[1] + y * cell + 1, cell - 2, cell - 2))

    if env.food is not None:
        block(env.food, (235, 90, 85))
    for i, position in enumerate(env.snake):
        block(position, (130, 240, 160) if i == 0 else (55, 170, 105))
    status = env.ending_reason or ("paused" if paused else "playing")
    lines = [
        (f"Snake | score {env.score} | steps {env.steps} | {status}", (18, 14)),
        ("A / Left: turn left    D / Right: turn right", (18, 548)),
        ("Space: pause    R: restart    Esc: quit", (18, 578)),
        (f"Without food: {env.steps_without_food}/{env.config.food_limit}", (18, 608)),
    ]
    for text, position in lines:
        screen.blit(font.render(text, True, (225, 230, 235)), position)
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=8, help="simulation steps per second (default: 8)")
    parser.add_argument("--smoke-test", action="store_true", help="draw and step three frames, then exit")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or not 0 < args.fps <= 60:
        parser.error("--fps must be in (0, 60]")

    env = SnakeEnv()
    env.reset(seed=args.seed)
    pygame.display.init()
    pygame.font.init()
    try:
        screen = pygame.display.set_mode((600, 640))
        pygame.display.set_caption("Snake — Milestone 1")
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()
        pending_action = 0
        paused = False
        elapsed = 0.0
        frames = 0
        running = True
        while running:
            dt = min(clock.tick(60) / 1000.0, 0.25)
            if args.smoke_test:
                dt = 1 / args.fps
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        env.reset()
                        pending_action, elapsed = 0, 0.0
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        pending_action, elapsed = 0, 0.0
                    elif not paused and pending_action == 0:
                        if event.key in (pygame.K_a, pygame.K_LEFT):
                            pending_action = 2
                        elif event.key in (pygame.K_d, pygame.K_RIGHT):
                            pending_action = 1
            if not running:
                break
            if not paused and env.ending_reason is None:
                elapsed += dt
                while elapsed >= 1 / args.fps and env.ending_reason is None:
                    env.step(pending_action)
                    pending_action = 0
                    elapsed -= 1 / args.fps
            draw(screen, font, env, paused)
            frames += 1
            if args.smoke_test and frames == 3:
                print(f"PASS: viewer drew {frames} frames and advanced {env.steps} steps.")
                running = False
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
