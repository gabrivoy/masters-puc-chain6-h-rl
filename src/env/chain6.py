"""
chain6.py

Houses the environment Chain6Env.
"""

import random

from enum import IntEnum
from typing import Literal

from pydantic import BaseModel


class StepResult(BaseModel):
    """Result of taking a step in the environment.

    Attributes:
        next_state: Next state (1..6).
        reward_ext: Extrinsic reward.
            - 0 in non-terminal steps.
            - 1/100 upon reaching terminal state 1 without visiting state 6.
            - 1 upon reaching terminal state 1 after visiting state 6.
        done: Whether the episode has ended.
    """

    next_state: int
    reward_ext: float
    done: bool


class ActualState(BaseModel):
    """Actual state representation for Chain6Env.

    Attributes:
        state: Current state (1..6).
        visited_s6: Whether state 6 has been visited in the episode.
    """

    state: Literal[1, 2, 3, 4, 5, 6]
    visited_s6: bool


class PossibleActions(IntEnum):
    """Possible actions in the Chain6Env."""

    LEFT = 0
    RIGHT = 1


class Chain6Env:
    """Chain6 Environment."""

    def __init__(self, right_probability: float = 0.5) -> None:
        """Constructor method."""
        self.state: ActualState = ActualState(state=2, visited_s6=False)
        self.right_probability = right_probability

    def reset(self) -> int:
        """Resets the environment to the start state 2 and returns the integer state."""
        self.state = ActualState(state=2, visited_s6=False)
        return self.state.state

    def step(self, action: PossibleActions) -> StepResult:
        """
        Takes a step in the environment based on the given action.

        Args:
            action: Action to take (PossibleActions.LEFT or PossibleActions.RIGHT).

        Returns:
            StepResult: Result of the step containing next state, extrinsic reward,
                and whether the episode has ended.
        """
        actual_state = self.state.state

        if action == PossibleActions.LEFT:
            # Move left deterministically
            next_state = max(1, actual_state - 1)

        elif action == PossibleActions.RIGHT:
            # Move right stochastically
            if random.random() < self.right_probability:
                next_state = min(6, actual_state + 1)
            else:
                next_state = max(1, actual_state - 1)
        else:
            raise ValueError("Invalid action.")

        self.state.state = next_state
        if next_state == 6:
            self.state.visited_s6 = True

        # If we reached the terminal state (1), determine reward
        done = next_state == 1

        reward_ext = 0.0
        if done:
            reward_ext = 1.0 if self.state.visited_s6 else 0.01

        return StepResult(
            next_state=next_state,
            reward_ext=reward_ext,
            done=done,
        )

    @staticmethod
    def goal_set() -> tuple[int, ...]:
        """Returns the subgoal set used by hierarchical control: states (3, 4, 5, 6)."""
        return (3, 4, 5, 6)
