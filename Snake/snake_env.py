import pygame
import random
import torch

from typing import ClassVar, List
from collections.abc import Iterator
from dataclasses import dataclass
from collections import deque
from enum import IntEnum
from itertools import islice

# PyGame Settings
WIDTH      : int = 900
HEIGHT     : int = 600
CELL       : int = 30
GRID_WIDTH : int = 2

BACKGROUND  : str = 'black'
GRID_COLOR  : str = 'gray'
APPLE_COLOR : str = 'red'
HEAD_COLOR  : str = 'blue'
BODY_COLOR  : str = 'green'

pygame.init()

display : pygame.Surface    = pygame.display.set_mode((WIDTH, HEIGHT))

@dataclass(slots=True, frozen=True)
class Vector:
    x : int
    y : int

    def in_bounds(self) -> bool:
        return 0 <= self.x < WIDTH and 0 <= self.y < HEIGHT
    
    def __iter__(self) -> Iterator[int]:
        yield self.x
        yield self.y

    def __add__(self, o: "Vector") -> "Vector":
        return Vector(self.x + o.x, self.y + o.y)

    def __sub__(self, o: "Vector") -> "Vector":
        return Vector(self.x - o.x, self.y - o.y)
    
    def __mul__(self, s: int) -> "Vector":
        return Vector(self.x * s, self.y * s)

    def __neg__(self) -> "Vector":
        return Vector(-self.x, -self.y)

    def dot(self, o: "Vector") -> int:
        return self.x * o.x + self.y * o.y

    def rot_cw(self) -> "Vector":     # Clockwise on screen
        return Vector(-self.y, self.x)

    def rot_ccw(self) -> "Vector":    # Counter-Clockwise on the Screen
        return Vector(self.y, -self.x)

class Direction(IntEnum):
    vec : Vector

    def __new__(cls, value : int, vec : Vector):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.vec = vec
        return obj

    LEFT  = (0, Vector(-1, 0))
    RIGHT = (1, Vector(1, 0))
    DOWN  = (2, Vector(0, 1))
    UP    = (3, Vector(0, -1))

class Action(IntEnum):
    FORWARD = 0
    LEFT    = 1
    RIGHT   = 2

# Snake Class
class Snake:
    __slots__ = ('body', 'dir')

    def __init__(self) -> None:
        self.body : deque[Vector]
        self.dir  : Vector

    def _draw_snake_body_part(self, color : str, vector_id : int) -> None:
        pygame.draw.rect(display, color, pygame.Rect(self.body[vector_id].x, self.body[vector_id].y, CELL, CELL))

    def did_hit_snake_body(self, next_head : Vector, tail_vacates : bool = False) -> bool:
        body = islice(self.body, max(0, len(self.body) - 1)) if tail_vacates else iter(self.body)
        return next_head in body
    
    def did_snake_die(self, next_head : Vector, tail_vacates : bool = False) -> bool:
        if (
            not next_head.in_bounds() or self.did_hit_snake_body(next_head, tail_vacates)
        ):
            return True
        return False

    def get_direction_vector(self) -> Vector:
        return self.dir
    
    def get_next_head(self) -> Vector:
        dx, dy = self.get_direction_vector()
        hx, hy = self.body[0]

        return Vector(hx + dx * CELL, hy + dy * CELL)
    
    def reset(self) -> None:
        self.body = deque([Vector(WIDTH // 2, HEIGHT // 2)])
        self.dir  = Direction.RIGHT.vec

    def set_direction(self, action : int) -> None:
        if action == Action.RIGHT.value:
            self.dir = self.dir.rot_cw()
        elif action == Action.LEFT.value:
            self.dir = self.dir.rot_ccw()

    def draw_snake(self) -> None:
        # Body is empty for one frame right after a length-1 snake dies; nothing to draw.
        if not self.body:
            return

        # Draw Head
        self._draw_snake_body_part(HEAD_COLOR, 0)

        # Draw Body
        for i in range(1, len(self.body)):
            self._draw_snake_body_part(BODY_COLOR, i)
            
    def move(self, ate_apple : bool) -> bool:
        next_head : Vector = self.get_next_head()

        # Pop Right, if it didn't eat an apple then it should increase by 1 unit so we just don't need to pop
        if not ate_apple:
            self.body.pop()

        if self.did_snake_die(next_head):
            return False
        
        # Add Left 
        self.body.appendleft(next_head)

        return True

# Apple Class
class Apple:
    __slots__ = ('position')

    ALL_CELLS : ClassVar[List[Vector]] = [
        Vector(x, y) 
        for x in range(0, WIDTH, CELL) 
        for y in range(0, HEIGHT, CELL)
    ]

    REJECTION_SAMPLING_ATTEMPTS : ClassVar[int] = 20

    def __init__(self) -> None:
        self.position : Vector

    def reset(self, occupied_positions : set[Vector]) -> None:
        # This brings it down to O(1) avg instead of O(n). Really doesn't matter -\_(:>)_/-
        for _ in range(self.REJECTION_SAMPLING_ATTEMPTS):
            candidate : Vector = random.choice(self.ALL_CELLS)

            if candidate not in occupied_positions:
                self.position = candidate
                return

        free : List[Vector] = [c for c in self.ALL_CELLS if c not in occupied_positions]

        if not free:
            raise RuntimeError(f"There does not exist any free Spots for the Apple.")

        self.position = random.choice(free)

    def draw_apple(self) -> None:
        pygame.draw.rect(display, APPLE_COLOR, pygame.Rect(self.position.x, self.position.y, CELL, CELL))

# Main Training Class
@dataclass(slots=True, frozen=True)
class Transition:
    observation : torch.Tensor
    reward      : float
    done        : bool


class SnakeEnv:
    MAX_STEPS_WITHOUT_APPLE : ClassVar[int] = 400
    DANGER_RAY_DISTANCE     : ClassVar[int] = 10
    OBS_SIZE                : ClassVar[int] = 3 * DANGER_RAY_DISTANCE + 4

    BASE_REWARD  : ClassVar[float] = -0.01 # This is to encourage the AI to take the shortest paths to the apples.
    APPLE_REWARD : ClassVar[float] =  10.0
    HIT_REWARD   : ClassVar[float] = -10.0

    def __init__(self) -> None:
        self.snake : Snake = Snake()
        self.apple : Apple = Apple()

        self.steps  : int

        self.reset()

    def _is_dangerous(self, head : Vector, dir : Vector, step : int) -> bool:
        next_head : Vector = head + dir * step * CELL

        return self.snake.did_snake_die(next_head, tail_vacates=next_head != self.apple.position)

    def _is_terminal(self, snake_died : bool, ate_apple : bool) -> bool:
        if ate_apple:
            self.steps = 0  
        else:
            self.steps += 1
        
        return snake_died or self.steps >= self.MAX_STEPS_WITHOUT_APPLE

    def _calculate_reward(self, ate_apple : bool, snake_died : bool) -> float:
        return self.BASE_REWARD + (ate_apple * self.APPLE_REWARD) + (snake_died * self.HIT_REWARD)
    
    def reset(self) -> torch.Tensor:
        self.snake.reset()
        self.apple.reset(set(self.snake.body.copy()))

        self.steps  = 0

        return self.get_obs()

    def get_obs(self) -> torch.Tensor:
        if not self.snake.body:
            return torch.zeros(self.OBS_SIZE, dtype=torch.float32)

        head       : Vector = self.snake.body[0]
        dir_vector : Vector = self.snake.get_direction_vector()

        # Rotation Vectors
        fwd   : Vector = dir_vector
        right : Vector = dir_vector.rot_cw()
        left  : Vector = dir_vector.rot_ccw()

        to_apple : Vector = self.apple.position - head

        return torch.tensor([
            *[
                self._is_dangerous(head, dir_vector, step=i)
                for i in range(1, self.DANGER_RAY_DISTANCE + 1)
                for dir_vector in [fwd, right, left]
            ],

            # Find the Apple to the head by using the dot product.
            to_apple.dot(dir_vector) > 0, # Apple is ahead.
            to_apple.dot(dir_vector) < 0, # Apple is behind.
            to_apple.dot(right)      > 0, # Apple is to my right.
            to_apple.dot(left)       > 0, # Apple is to my left.
        ], dtype=torch.float32)

    def draw_scene(self) -> None:
        # Draw the Scene: Background -> Cells -> Apple -> Snake
        display.fill(BACKGROUND)

        # Draw the Grid
        for x in range(0, WIDTH, CELL):
            for y in range(0, HEIGHT, CELL):
                pygame.draw.rect(display, GRID_COLOR, (x, y, CELL, CELL), GRID_WIDTH)

        self.apple.draw_apple()
        self.snake.draw_snake()

    def step(self, action : int) -> Transition:
        """
        actions:
            0 = Forward
            1 = Turn Left
            2 = Turn Right
        returns Transition(
            Observation=torch.Tensor([...]), reward=float(...), done=bool(...)
        )
        """
        self.snake.set_direction(action)

        ate_apple   : bool = (self.apple.position == self.snake.get_next_head())
        snake_alive : bool = self.snake.move(ate_apple)
        snake_died  : bool = not snake_alive

        if ate_apple and snake_alive:
            self.apple.reset(set(self.snake.body))

        done   : bool  = self._is_terminal(snake_died, ate_apple)
        reward : float = self._calculate_reward(ate_apple, snake_died)

        return Transition(
            observation = self.get_obs(),
            reward      = reward,
            done        = done
        )

        
        






