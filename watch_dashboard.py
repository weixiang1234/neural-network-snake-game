"""Interactive Snake DQN dashboard with real activations and optional safety lookahead."""

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from checkpoints import load_inference
from observations import encode_observation, observation_order
from safety_lookahead import select_action
from snake_env import SnakeEnv


WIDTH, HEIGHT = 1450, 930
BOARD_X, BOARD_Y, BOARD_SIZE = 36, 148, 500
INPUT_X, HIDDEN_X, OUTPUT_X = 843, 1086, 1273
NET_TOP, NET_BOTTOM = 188, 619
OUTPUT_YS = (243, 393, 543)

BG = (14, 20, 29)
PANEL = (23, 32, 43)
PANEL_LINE = (47, 61, 75)
TEXT = (233, 239, 244)
MUTED = (155, 172, 185)
FAINT = (95, 113, 129)
GREEN = (105, 233, 164)
TEAL = (85, 197, 198)
ORANGE = (239, 162, 106)
RED = (244, 109, 108)
GOLD = (248, 201, 105)
PURPLE = (177, 147, 244)
ACTIONS = ("STRAIGHT", "RIGHT", "LEFT")


def spaced_y(count):
    if count == 1:
        return [(NET_TOP + NET_BOTTOM) // 2]
    return [round(NET_TOP + i * (NET_BOTTOM - NET_TOP) / (count - 1))
            for i in range(count)]


def training_curve(path):
    """Return (training episode, greedy mean score) from an evaluation CSV."""
    if path is None:
        return []
    groups = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"training_episode", "score"}.issubset(reader.fieldnames):
            raise ValueError("evaluation CSV requires training_episode and score columns")
        for row in reader:
            episode = int(row["training_episode"])
            groups.setdefault(episode, []).append(int(row["score"]))
    return [(ep, sum(values) / len(values)) for ep, values in sorted(groups.items())]


def inspect_policy(env, model, horizon):
    """Compute actual inference values and a non-mutating safety decision."""
    features = encode_observation(env, model.config.input_size)
    activations = model.activations(features[None, :])
    q_values = activations["q_values"][0]
    decision = select_action(env, q_values, future_steps=horizon)
    return features, activations["hidden"][0], q_values, decision


def paint_text(screen, font, value, x, y, color=TEXT):
    screen.blit(font.render(str(value), True, color), (x, y))


def panel(screen, rect):
    pygame.draw.rect(screen, PANEL, rect, border_radius=12)
    pygame.draw.rect(screen, PANEL_LINE, rect, width=1, border_radius=12)


def draw_board(screen, fonts, env, game, preview):
    panel(screen, pygame.Rect(18, 118, 556, 570))
    paint_text(screen, fonts["title"], "SNAKE ENVIRONMENT", 36, 126)
    cell = max(4, min(25, BOARD_SIZE // max(env.config.width, env.config.height)))
    span_x, span_y = env.config.width * cell, env.config.height * cell
    start_x = BOARD_X + (BOARD_SIZE - span_x) // 2
    start_y = BOARD_Y + (BOARD_SIZE - span_y) // 2
    pygame.draw.rect(screen, (18, 27, 36), pygame.Rect(start_x, start_y, span_x, span_y))
    for i in range(env.config.width + 1):
        x = start_x + i * cell
        pygame.draw.line(screen, (37, 48, 59), (x, start_y), (x, start_y + span_y))
    for i in range(env.config.height + 1):
        y = start_y + i * cell
        pygame.draw.line(screen, (37, 48, 59), (start_x, y), (start_x + span_x, y))

    def tile(position, color):
        x, y = position
        pygame.draw.rect(screen, color,
                         pygame.Rect(start_x + x * cell + 2, start_y + y * cell + 2,
                                     max(1, cell - 3), max(1, cell - 3)),
                         border_radius=min(4, cell // 5))

    if env.food is not None:
        tile(env.food, RED)
    for index, position in enumerate(reversed(env.snake)):
        tile(position, GREEN if index == len(env.snake) - 1 else (56, 159, 116))
    pygame.draw.rect(screen, (81, 99, 109), (start_x, start_y, span_x, span_y), 2)

    status = env.ending_reason or ("PAUSED" if game["paused"] else "PLAYING")
    paint_text(screen, fonts["small"], f"Status: {status}", 37, 657, GOLD if env.ending_reason else MUTED)
    paint_text(screen, fonts["small"],
               f"Last move: {ACTIONS[game['last_action']] if game['last_action'] is not None else '—'}",
               292, 657, TEXT)


def important_edges(model, ys_input, ys_hidden):
    """Display a sparse subset of actual learned weights for readability."""
    w1, w2 = model.params["W1"], model.params["W2"]
    selected = []
    for j in range(w1.shape[1]):
        for i in np.argsort(np.abs(w1[:, j]))[-1:]:
            selected.append(((INPUT_X, ys_input[int(i)]), (HIDDEN_X, ys_hidden[j]), float(w1[i, j])))
    for k in range(w2.shape[1]):
        for j in np.argsort(np.abs(w2[:, k]))[-min(4, w2.shape[0]):]:
            selected.append(((HIDDEN_X, ys_hidden[int(j)]), (OUTPUT_X, OUTPUT_YS[k]), float(w2[j, k])))
    return selected


def draw_network(screen, fonts, model, preview, edges, horizon, current_seed, ended):
    panel(screen, pygame.Rect(591, 118, 842, 570))
    size = model.config.input_size
    paint_text(screen, fonts["title"], f"LIVE Q NETWORK  |  {size}  →  {model.config.hidden_size}  →  3", 612, 126)
    paint_text(screen, fonts["tiny"], "INPUT FEATURES", 625, 160, MUTED)
    paint_text(screen, fonts["tiny"], "ReLU ACTIVATIONS", 1031, 160, MUTED)
    paint_text(screen, fonts["tiny"], "ACTION Q-VALUES", 1249, 160, MUTED)
    if ended:
        paint_text(screen, fonts["tiny"], "FROZEN", 1353, 135, GOLD)
    if preview is None:
        paint_text(screen, fonts["small"], "Game finished — showing final board", 685, 392, MUTED)
        return

    features, hidden, q_values, decision = preview
    ys_input = spaced_y(size)
    ys_hidden = spaced_y(model.config.hidden_size)

    for first, second, weight in edges:
        color = (42, 96, 101) if weight >= 0 else (103, 60, 65)
        pygame.draw.aaline(screen, color, first, second)

    labels = observation_order(size)
    for i, (value, y) in enumerate(zip(features, ys_input)):
        label = labels[i].replace("_", " ")
        paint_text(screen, fonts["tiny"], label, 613, y - 9, TEXT if value else MUTED)
        paint_text(screen, fonts["tiny"], f"{value:+.2f}" if size == 13 and i >= 11 else str(int(value)),
                   793, y - 9, GOLD if value else FAINT)
        color = TEAL if value > 0 else PANEL_LINE
        pygame.draw.circle(screen, color, (INPUT_X, y), 7)
        pygame.draw.circle(screen, (195, 220, 226), (INPUT_X, y), 7, 1)

    max_activation = max(float(hidden.max(initial=0)), 1e-9)
    for value, y in zip(hidden, ys_hidden):
        fraction = max(0.0, min(1.0, float(value) / max_activation))
        color = (36 + round(fraction * 120), 60 + round(fraction * 170),
                 80 + round(fraction * 120))
        pygame.draw.circle(screen, color, (HIDDEN_X, y), 6)
        pygame.draw.circle(screen, (83, 111, 123), (HIDDEN_X, y), 6, 1)

    for idx, y in enumerate(OUTPUT_YS):
        is_greedy = idx == decision.greedy_action
        is_final = idx == decision.action
        fill = GREEN if is_final else GOLD if is_greedy else (69, 87, 104)
        pygame.draw.circle(screen, fill, (OUTPUT_X, y), 11)
        pygame.draw.circle(screen, BG, (OUTPUT_X, y), 11, 2)
        paint_text(screen, fonts["small"], ACTIONS[idx], 1294, y - 21,
                   GREEN if is_final else TEXT)
        paint_text(screen, fonts["body"], f"{float(q_values[idx]):+.2f}", 1294, y + 1,
                   GREEN if is_final else MUTED)
        if is_greedy:
            paint_text(screen, fonts["tiny"], "Q MAX", 1371, y - 20, GOLD)

    timing = "LAST" if ended else "NEXT"
    explanation = (f"{timing}: {ACTIONS[decision.action]}  |  DQN: {ACTIONS[decision.greedy_action]}"
                   f"  |  Override: {'YES' if decision.overridden else 'NO'}")
    paint_text(screen, fonts["small"], explanation, 614, 648,
               ORANGE if decision.overridden else TEXT)
    if decision.no_viable_action:
        paint_text(screen, fonts["tiny"], "No survivable continuation found within horizon", 613, 669, RED)
    else:
        paint_text(screen, fonts["tiny"],
                   "Edges: top 1 per hidden + 4 per output  |  teal: +weight  red: −weight  |  hidden: relative",
                   613, 669, MUTED)


def draw_chart(screen, fonts, curve, history, is_training):
    panel(screen, pygame.Rect(18, 703, 1415, 177))
    title = "TRAINING GREEDY EVALUATION" if is_training else "COMPLETED EPISODES IN THIS SESSION"
    paint_text(screen, fonts["title"], title, 35, 710)
    paint_text(screen, fonts["tiny"],
               "Mean score / checkpoint (development seeds)" if is_training else "One score per completed game",
               749, 713, MUTED)
    points = curve if is_training else history
    if not points:
        paint_text(screen, fonts["body"], "No data yet. Finish an episode or pass --evaluation-csv.",
                   464, 782, MUTED)
        return

    x0, y0, x1, y1 = 98, 752, 1388, 846
    pygame.draw.line(screen, PANEL_LINE, (x0, y0), (x0, y1))
    pygame.draw.line(screen, PANEL_LINE, (x0, y1), (x1, y1))
    xs = [float(pt[0]) for pt in points]
    ys = [float(pt[1]) for pt in points]
    low_x, high_x = min(xs), max(xs)
    max_y = max(max(ys) * 1.15, 1.0)
    for index in range(4):
        py = round(y1 - index * (y1 - y0) / 3)
        pygame.draw.line(screen, (38, 50, 62), (x0, py), (x1, py))
        paint_text(screen, fonts["tiny"], f"{max_y * index / 3:.1f}", 46, py - 9, MUTED)

    def coords(x, y):
        px = x0 + (x - low_x) / (high_x - low_x) * (x1 - x0) if high_x > low_x else (x0 + x1) / 2
        py = y1 - y / max_y * (y1 - y0)
        return round(px), round(py)

    plotted = [coords(x, y) for x, y in points]
    if len(plotted) >= 2:
        pygame.draw.lines(screen, TEAL, False, plotted, 2)
    for xy in plotted:
        pygame.draw.circle(screen, GREEN, xy, 4)
    paint_text(screen, fonts["tiny"], str(int(low_x)), x0, 853, MUTED)
    paint_text(screen, fonts["tiny"], str(int(high_x)), x1 - 28, 853, MUTED)
    paint_text(screen, fonts["small"], f"Last: {ys[-1]:.2f}", 1284, 713, GREEN)


def draw(screen, fonts, env, model, preview, edges, game, curve, history, use_curve):
    screen.fill(BG)
    paint_text(screen, fonts["header"], "SNAKE   /   AUTONOMOUS AGENT LAB", 26, 15, TEXT)
    paint_text(screen, fonts["small"],
               f"EPISODE {game['episode']}   •   SEED {game['seed']}   •   SCORE {env.score}"
               f"   •   STEPS {env.steps}   •   NO FOOD {env.steps_without_food}/{env.config.food_limit}",
               28, 75, MUTED)
    paint_text(screen, fonts["body"], f"LOOKAHEAD {game['mode']}   |   OVERRIDES {game['overrides']}",
               1095, 73, GOLD if game["mode"] else MUTED)
    draw_board(screen, fonts, env, game, preview)
    draw_network(screen, fonts, model, preview, edges, game["mode"], game["seed"], env.ending_reason is not None)
    draw_chart(screen, fonts, curve, history, use_curve)
    paint_text(screen, fonts["small"],
               "SPACE pause    R restart    N next seed    0/1/2 select controller    +/- speed    S screenshot    ESC quit",
               29, 897, TEXT)
    speed_text = f"{game['speed']:g} moves/s"
    speed_x = WIDTH - 25 - fonts["small"].size(speed_text)[0]
    paint_text(screen, fonts["small"], speed_text, speed_x, 897, GOLD)
    pygame.display.flip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--lookahead", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--speed", type=float, default=12.0, help="simulation moves per second (1-60)")
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--evaluation-csv", type=Path, help="optional training evaluation.csv")
    parser.add_argument("--screenshot", type=Path, default=Path("results/snake_dashboard.png"))
    parser.add_argument("--smoke-test", action="store_true", help="draw and run 3 game steps, then exit")
    args = parser.parse_args(argv)
    if args.seed < 0 or not math.isfinite(args.speed) or not 1 <= args.speed <= 60:
        parser.error("--seed must be nonnegative and speed in [1, 60]")
    if not math.isfinite(args.hold_seconds) or args.hold_seconds < 0:
        parser.error("--hold-seconds must be finite and nonnegative")

    model, config = load_inference(args.checkpoint)
    try:
        curve = training_curve(args.evaluation_csv)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    env = SnakeEnv(config.environment)
    env.reset(seed=args.seed)
    ys_input = spaced_y(model.config.input_size)
    ys_hidden = spaced_y(model.config.hidden_size)
    edges = important_edges(model, ys_input, ys_hidden)
    history = []
    game = dict(episode=1, seed=args.seed, mode=args.lookahead, paused=False,
                last_action=None, overrides=0, speed=args.speed)
    last_preview = None
    preview_cache_key = None
    preview = None
    elapsed = ended_for = 0.0

    pygame.display.init()
    pygame.font.init()
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake AI — DQN and Safety Lookahead")
        fonts = {"header": pygame.font.Font(None, 33), "title": pygame.font.Font(None, 25),
                 "body": pygame.font.Font(None, 25), "small": pygame.font.Font(None, 21),
                 "tiny": pygame.font.Font(None, 19)}
        clock = pygame.time.Clock()

        def restart(next_seed):
            nonlocal elapsed, ended_for, preview_cache_key, preview, last_preview
            if next_seed:
                game["seed"] += 1
                game["episode"] += 1
            env.reset(seed=game["seed"])
            game["last_action"] = None
            game["overrides"] = 0
            elapsed = ended_for = 0.0
            preview_cache_key = None
            preview = last_preview = None

        def current_preview():
            nonlocal preview_cache_key, preview
            if env.ending_reason is not None:
                return last_preview
            key = (game["seed"], env.steps, game["mode"])
            if key != preview_cache_key:
                preview = inspect_policy(env, model, game["mode"])
                preview_cache_key = key
            return preview

        frames = 0
        running = True
        while running:
            dt = min(clock.tick(60) / 1000, 0.25)
            if args.smoke_test:
                dt = 1.0 / game["speed"]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        game["paused"] = not game["paused"]
                        elapsed = 0.0
                    elif event.key == pygame.K_r:
                        restart(False)
                    elif event.key == pygame.K_n:
                        restart(True)
                    elif event.key in (pygame.K_0, pygame.K_1, pygame.K_2):
                        game["mode"] = {pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2}[event.key]
                        preview_cache_key = None
                        elapsed = 0.0
                    elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                        game["speed"] = min(60.0, game["speed"] * 2)
                        elapsed = 0.0
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        game["speed"] = max(1.0, game["speed"] / 2)
                        elapsed = 0.0
                    elif event.key == pygame.K_s:
                        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                        pygame.image.save(screen, str(args.screenshot))
                        print(f"Saved screenshot: {args.screenshot}")
            if not running:
                break

            if not game["paused"] and env.ending_reason is None:
                elapsed += dt
                while elapsed >= 1.0 / game["speed"] and env.ending_reason is None:
                    last_preview = current_preview()
                    decision = last_preview[3]
                    env.step(decision.action)
                    game["last_action"] = decision.action
                    game["overrides"] += int(decision.overridden)
                    preview_cache_key = None
                    elapsed -= 1.0 / game["speed"]
                    if env.ending_reason is not None:
                        history.append((game["episode"], env.score))
                        print(f"episode={game['episode']} seed={game['seed']} lookahead={game['mode']} "
                              f"score={env.score} steps={env.steps} end={env.ending_reason} "
                              f"overrides={game['overrides']}", flush=True)
                        ended_for = 0.0
            elif not game["paused"] and env.ending_reason is not None:
                ended_for += dt
                if ended_for >= args.hold_seconds and not args.smoke_test:
                    restart(True)

            shown_preview = current_preview()
            draw(screen, fonts, env, model, shown_preview, edges, game, curve, history, bool(args.evaluation_csv))
            frames += 1
            if args.smoke_test and frames >= 3:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(screen, str(args.screenshot))
                print(f"PASS: loaded policy, rendered {frames} frames, steps={env.steps}; "
                      f"lookahead={game['mode']}; screenshot={args.screenshot}")
                break
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
