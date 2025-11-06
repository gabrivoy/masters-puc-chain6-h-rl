"""
q_learning.py

Tabular Q-learning agent for the Chain6 environment.

This agent learns a single action-value function Q(s, a) with
epsilon-greedy exploration and a standard one-step TD(0) update.
"""

import random

import numpy as np

from env.chain6 import Chain6Env, PossibleActions


class FlatQLearning:
    """Tabular Q-learning baseline for Chain6.

    The agent maintains a table Q[s, a] indexed by state in {1..6} and action
    in {LEFT, RIGHT}. Row 0 is kept unused for alignment with environment
    states (1-based). Action selection is epsilon-greedy with a linear
    annealing schedule ε_t ∈ [eps_start, eps_end].

    The TD target for a transition (s, a, r, s') is:
        y = r + gamma * max_a' Q(s', a')  | if not terminal
        y = r                             | if terminal

    and the update is:
        Q(s, a) = Q(s, a) + alpha * (y - Q(s, a)).

    Attributes:
        q_value_table: Action-value table with shape (7, 2); index 0 is unused.
        learning_rate: Scalar alpha controlling the step size of the update.
        discount_factor: Scalar gamma discounting future returns.
        epsilon_start: Initial epsilon for epsilon-greedy exploration.
        epsilon_end: Final epsilon after the annealing schedule finishes.
        epsilon_steps: Number of selections over which epsilon anneals from
            start to end.
        exploration_step_count: Internal counter for annealing progress.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.1,
        eps_steps: int = 5_000,
    ) -> None:
        self.q_value_table: np.ndarray = np.zeros((7, 2), dtype=float)

        self.learning_rate: float = alpha
        self.discount_factor: float = gamma

        self.epsilon_start: float = eps_start
        self.epsilon_end: float = eps_end
        self.epsilon_steps: int = max(1, eps_steps)

        self.exploration_step_count: int = 0

    def _current_epsilon(self) -> float:
        """Returns the current epsilon according to a linear schedule.

        The schedule interpolates linearly from `epsilon_start` to `epsilon_end`
        across `epsilon_steps` calls, and then stays at `epsilon_end`.

        Returns:
            Current epsilon value for epsilon-greedy selection.
        """
        progress_fraction = min(1.0, self.exploration_step_count / self.epsilon_steps)
        return self.epsilon_start + progress_fraction * (
            self.epsilon_end - self.epsilon_start
        )

    def select_action(self, state: int) -> int:
        """Selects an action via epsilon-greedy over Q(state, ·).

        Args:
            state: Current environment state in {1..6}.

        Returns:
            Integer-encoded action compatible with `PossibleActions`.
        """
        epsilon = self._current_epsilon()
        self.exploration_step_count += 1

        if random.random() < epsilon:
            return random.choice([PossibleActions.LEFT, PossibleActions.RIGHT]).value

        return int(np.argmax(self.q_value_table[state]))

    def run_episode(self, env: Chain6Env, max_steps: int = 200) -> float:
        """Runs a single episode in the environment.

        Args:
            env: Instance of Chain6Env.
            max_steps: Upper bound on steps to avoid pathological loops.

        Returns:
            Sum of extrinsic rewards collected during the episode.
        """
        state: int = env.reset()
        episode_extrinsic_return: float = 0.0

        for _ in range(max_steps):
            action_int: int = self.select_action(state)
            step = env.step(PossibleActions(action_int))

            # TD target with extrinsic reward (often 0 in this sparse setting).
            if step.done:
                td_target: float = step.reward_ext
            else:
                td_target = step.reward_ext + self.discount_factor * float(
                    np.max(self.q_value_table[step.next_state])
                )

            # Q-learning update
            self.q_value_table[state, action_int] += self.learning_rate * (
                td_target - self.q_value_table[state, action_int]
            )

            state = step.next_state
            episode_extrinsic_return += step.reward_ext

            if step.done:
                break

        return episode_extrinsic_return
