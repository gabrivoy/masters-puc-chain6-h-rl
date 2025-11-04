"""
h_q_learning.py

Houses the HierarchicalTabular agent for the Chain6Env environment.
"""

import random

import numpy as np

from env.chain6 import Chain6Env, PossibleActions


class EpsilonSched:
    def __init__(self, start: float, end: float, steps: int):
        self.start, self.end, self.steps, self.t = start, end, max(1, steps), 0

    def value(self) -> float:
        frac = min(1.0, self.t / self.steps)
        return self.start + frac * (self.end - self.start)

    def step(self) -> None:
        self.t += 1


class HierarchicalTabular:
    def __init__(
        self,
        goals: tuple[int, ...],
        gamma1: float = 0.99,
        gamma2: float = 0.99,
        alpha1: float = 0.2,
        alpha2: float = 0.1,
        timeout_H: int = 10,
    ):
        self.goals = goals
        self.gamma1, self.gamma2 = gamma1, gamma2
        self.alpha1, self.alpha2 = alpha1, alpha2
        self.timeout_H = timeout_H
        # Q1: (state, action, goal_index), Q2: (state, goal_index)
        self.Q1 = np.zeros((7, 2, len(goals)), dtype=float)
        self.Q2 = np.zeros((7, len(goals)), dtype=float)
        self.goal_to_idx = {g: i for i, g in enumerate(goals)}
        self.meta_eps = EpsilonSched(1.0, 0.1, 5000)
        self.ctrl_eps = {g: EpsilonSched(1.0, 0.1, 1000) for g in goals}

    def select_goal(self, s: int) -> int:
        eps = self.meta_eps.value()
        self.meta_eps.step()
        if random.random() < eps:
            return random.choice(self.goals)
        g_idx = int(np.argmax(self.Q2[s]))
        return self.goals[g_idx]

    def select_action(self, s: int, g: int) -> int:
        eps = self.ctrl_eps[g].value()
        self.ctrl_eps[g].step()
        if random.random() < eps:
            return random.choice([PossibleActions.LEFT, PossibleActions.RIGHT]).value
        gi = self.goal_to_idx[g]
        return int(np.argmax(self.Q1[s, :, gi]))

    def update_controller(
        self, s: int, a: int, r_int: float, s_next: int, g: int
    ) -> None:
        gi = self.goal_to_idx[g]
        q_sa = self.Q1[s, a, gi]
        td = r_int + self.gamma1 * float(np.max(self.Q1[s_next, :, gi]))
        self.Q1[s, a, gi] = q_sa + self.alpha1 * (td - q_sa)

    def update_meta(
        self, s_pre: int, g: int, R_ext: float, s_pos: int, tau: int, done: bool
    ) -> None:
        gi = self.goal_to_idx[g]
        q_sg = self.Q2[s_pre, gi]
        if done:
            target = R_ext
        else:
            target = R_ext + (self.gamma2 ** max(1, tau)) * float(
                np.max(self.Q2[s_pos])
            )
        self.Q2[s_pre, gi] = q_sg + self.alpha2 * (target - q_sg)

    def run_episode(self, env: Chain6Env) -> float:
        s = env.reset()
        total_ext = 0.0
        while True:
            g = self.select_goal(s)
            s_pre, R_ext, tau = s, 0.0, 0
            step = None  # for type clarity
            for _ in range(self.timeout_H):
                a = self.select_action(s, g)
                step = env.step(PossibleActions(a))
                r_int = 1.0 if (step.next_state == g) else 0.0
                self.update_controller(s, a, r_int, step.next_state, g)
                R_ext += step.reward_ext
                tau += 1
                s = step.next_state
                if step.next_state == g or step.done:
                    break
            # step is guaranteed to be set because timeout_H >= 1
            self.update_meta(s_pre, g, R_ext, s, tau, done=bool(step.done))
            if step.done:
                total_ext += R_ext
                break
        return total_ext
