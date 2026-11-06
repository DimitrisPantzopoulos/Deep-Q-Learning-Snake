import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from dataclasses import dataclass
from typing import ClassVar, List

@dataclass(slots=True, frozen=True)
class Memory:
    state      : torch.Tensor
    action     : int
    reward     : float
    next_state : torch.Tensor
    done       : bool

class SnakeAI(nn.Module):
    MAX_MEMORY_LEN       : ClassVar[int]   = 10_000
    EXPLORATION_FRACTION : ClassVar[float] = 0.5

    def __init__(self, n_episodes : int | None = None, n_obs : int=19, n_actions : int=3, hidden : int=32) -> None:
        super().__init__()

        self.layers : nn.Sequential = nn.Sequential(
            nn.Linear(n_obs, hidden),     nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, n_actions),        
        )

        self.memory :  deque[Memory] = deque(maxlen=self.MAX_MEMORY_LEN)

        self.optimizer : optim.AdamW  = optim.AdamW(self.parameters(), lr=1e-3)
        self.loss_fn   : nn.HuberLoss = nn.HuberLoss()

        self.gamma      : float = 0.9
        self.batch_size : int   = 64

        self.epsilon       : float = 1.0
        self.epsilon_min   : float = 0.02

        decay_episodes : int = max(1, int(n_episodes * self.EXPLORATION_FRACTION)) if n_episodes is not None else 0
        self.epsilon_decay : float = self.epsilon_min ** (1 / decay_episodes) if n_episodes is not None else 0

    def forward(self, obs : torch.Tensor) -> torch.Tensor:
        return self.layers(obs)

    def sample_actions(self, state : torch.Tensor) -> int:
        if random.random() < self.epsilon:
            return random.randrange(3)

        with torch.no_grad():
            return int(self.forward(state).argmax().item())

    def store_memory(
            self,
            state      : torch.Tensor,
            action     : int,
            reward     : float,
            next_state : torch.Tensor,
            done       : bool,
        ) -> None:
        self.memory.append(Memory(
            state, action, reward, next_state, done
        ))

    def learn(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        batch : List[Memory] = random.sample(self.memory, self.batch_size)

        states      : torch.Tensor = torch.stack([m.state   for m in batch])
        actions     : torch.Tensor = torch.tensor([m.action for m in batch], dtype=torch.int64).unsqueeze(1)
        rewards     : torch.Tensor = torch.tensor([m.reward for m in batch], dtype=torch.float32).unsqueeze(1)
        next_states : torch.Tensor = torch.stack([m.next_state for m in batch])
        dones       : torch.Tensor = torch.tensor([m.done for m in batch], dtype=torch.float32).unsqueeze(1)

        current_q_values : torch.Tensor = self(states).gather(1, actions)

        with torch.no_grad():
            next_q_values   : torch.Tensor = self(next_states).max(1, keepdim=True).values
            target_q_values : torch.Tensor = rewards + self.gamma * next_q_values * (1 - dones)

        loss : torch.Tensor = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
