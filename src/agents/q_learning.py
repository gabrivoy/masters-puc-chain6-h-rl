"""
q_learning.py

Houses the FlatQLearning agent for the Chain6Env environment.
"""

import random

import numpy as np

from env.chain6 import Chain6Env, PossibleActions


class FlatQLearning:
    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.1,
        eps_steps: int = 5000,
    ):
        # States indexed 1..6; keep row 0 unused for simplicity
        self.Q = np.zeros((7, 2), dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = max(1, eps_steps)
        self.t = 0

    def _eps(self) -> float:
        frac = min(1.0, self.t / self.eps_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, s: int) -> int:
        eps = self._eps()
        self.t += 1
        if random.random() < eps:
            return random.choice([PossibleActions.LEFT, PossibleActions.RIGHT]).value
        return int(np.argmax(self.Q[s]))

    def run_episode(self, env: Chain6Env, max_steps: int = 200) -> float:
        s = env.reset()
        ret = 0.0
        for _ in range(max_steps):
            a = self.select_action(s)
            step = env.step(PossibleActions(a))
            # TD target with extrinsic reward (often 0)
            target = step.reward_ext + self.gamma * (
                0.0 if step.done else float(np.max(self.Q[step.next_state]))
            )
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])
            s = step.next_state
            ret += step.reward_ext
            if step.done:
                break
        return ret
