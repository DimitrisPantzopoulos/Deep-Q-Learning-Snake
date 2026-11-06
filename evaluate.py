from __future__ import annotations

import argparse
import os
import statistics
import sys

import pygame
import torch

from pathlib import Path
from typing import TYPE_CHECKING

from Model.model import SnakeAI

if TYPE_CHECKING:
    from Snake.snake_env import SnakeEnv, Transition

torch.set_num_threads(1)

DEFAULT_CHECKPOINT : Path = Path(__file__).parent / "checkpoints" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake AI checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"checkpoint to evaluate (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="show the game window while evaluating",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="rendered frame rate (default: 60)",
    )
    args: argparse.Namespace = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    return args


def handle_pygame_events() -> None:
    for event in pygame.event.get():
        if event.type in (pygame.QUIT, pygame.WINDOWCLOSE):
            pygame.quit()
            raise SystemExit


def run_episode(
    model  : SnakeAI,
    env    : SnakeEnv,
    render : bool,
    fps    : int,
    clock  : pygame.time.Clock,
) -> tuple[float, int]:
    state: torch.Tensor = env.reset()
    episode_reward: float = 0.0
    episode_steps: int = 0
    done: bool = False

    while not done:
        handle_pygame_events()
        action: int = model.sample_actions(state)
        transition: Transition = env.step(action)

        state = transition.observation
        done = transition.done
        episode_reward += transition.reward
        episode_steps += 1

        if render:
            env.draw_scene()
            pygame.display.flip()
            clock.tick(fps)

    return episode_reward, episode_steps


def main() -> None:
    args : argparse.Namespace = parse_args()
    if not args.render:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    from Snake.snake_env import SnakeEnv

    device : torch.device = torch.device("cpu")

    env   : SnakeEnv = SnakeEnv()
    model : SnakeAI  = SnakeAI(n_obs=env.OBS_SIZE)

    state_dict : dict[str, torch.Tensor] = torch.load(
        args.checkpoint,
        map_location = device,
        weights_only = True,
    )

    model.load_state_dict(state_dict)
    model.epsilon = 0.0
    model.eval()

    clock   : pygame.time.Clock = pygame.time.Clock()
    rewards : list[float] = []
    steps   : list[int] = []

    try:
        for episode in range(1, args.episodes + 1):
            episode_reward, episode_steps = run_episode(
                model, env, args.render, args.fps, clock
            )
            rewards.append(episode_reward)
            steps.append(episode_steps)
            print(
                f"Episode {episode}/{args.episodes}: "
                f"reward={episode_reward:.2f}, steps={episode_steps}"
            )
    finally:
        pygame.quit()

    print(
        f"Summary ({args.episodes} episodes): "
        f"mean reward={statistics.fmean(rewards):.2f} "
        f"(std={statistics.pstdev(rewards):.2f}), "
        f"mean steps={statistics.fmean(steps):.1f}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(130)
