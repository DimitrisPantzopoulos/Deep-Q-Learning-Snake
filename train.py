import pygame
import torch

from pathlib import Path
from tqdm import tqdm
from Model.model import SnakeAI
from Snake.snake_env import SnakeEnv, Transition

torch.set_num_threads(1)

TRAINING_EPISODES    : int   = 20_000
EVAL_EPISODES        : int   = 100
RENDER_DURING_TRAIN  : bool  = False
FPS                  : int   = 60
CHECKPOINT_EVERY     : int   = 2_000
CHECKPOINT_DIR       : Path  = Path(__file__).parent / "checkpoints"

env   : SnakeEnv = SnakeEnv()
model : SnakeAI  = SnakeAI(TRAINING_EPISODES, n_obs=env.OBS_SIZE)

clock : pygame.time.Clock = pygame.time.Clock()

CHECKPOINT_DIR.mkdir(exist_ok=True)

def handle_pygame_events() -> None:
    for event in pygame.event.get():
        if event.type in (pygame.QUIT, pygame.WINDOWCLOSE):
            pygame.quit()
            raise SystemExit

def run_episode(render : bool, learn : bool) -> tuple[float, int]:
    state : torch.Tensor = env.reset()
    done  : bool  = False

    episode_reward : float = 0.0
    episode_steps  : int   = 0

    while not done:
        handle_pygame_events()

        action : int = model.sample_actions(state)

        transition : Transition = env.step(action)

        if learn:
            model.store_memory(state, action, transition.reward, transition.observation, transition.done)
            model.learn()

        state = transition.observation
        done  = transition.done

        episode_reward += transition.reward
        episode_steps  += 1

        if render:
            env.draw_scene()
            pygame.display.flip()
            clock.tick(FPS)

    return episode_reward, episode_steps


best_reward : float = float("-inf")
progress    : tqdm  = tqdm(range(TRAINING_EPISODES), desc="Training")

for episode in progress:
    episode_reward, episode_steps = run_episode(render=RENDER_DURING_TRAIN, learn=True)

    model.update_epsilon()

    if episode_reward > best_reward:
        best_reward = episode_reward
        torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")

    progress.set_postfix(
        reward  = f"{episode_reward:.1f}",
        steps   = episode_steps,
        epsilon = f"{model.epsilon:.3f}",
        best    = f"{best_reward:.1f}",
    )

    if (episode + 1) % CHECKPOINT_EVERY == 0:
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"checkpoint_{episode + 1}.pt")

torch.save(model.state_dict(), CHECKPOINT_DIR / "final.pt")

# Evaluation phase: always rendered, greedy policy (no exploration), no further learning.
model.epsilon = 0.0

for episode in range(EVAL_EPISODES):
    episode_reward, episode_steps = run_episode(render=True, learn=False)
    print(f"Eval episode {episode + 1}/{EVAL_EPISODES}: reward={episode_reward:.1f} steps={episode_steps}")

pygame.quit()