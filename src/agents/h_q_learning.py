"""
h_q_learning.py

Hierarchical tabular agent for Chain6 (meta-controller + controller).

The agent factorizes decision-making into:
  - a meta-controller Q2(s, g) that selects a sub-goal g ∈ {3,4,5,6};
  - a controller Q1_g(s, a) that acts to reach the chosen sub-goal g,
    receiving an intrinsic reward r_int = 1 when s == g, else 0.

This decomposes a long-horizon sparse-reward problem into shorter-horizon
goal-reaching problems, easing exploration toward the informative region S6.
"""

import random

import numpy as np

from env.chain6 import Chain6Env, PossibleActions


class EpsilonSchedule:
    """Linear epsilon schedule.

    Attributes:
        start: Initial epsilon.
        end: Final epsilon after annealing.
        steps: Number of steps over which epsilon anneals linearly.
        t: Internal counter of how many values were consumed.
    """

    def __init__(self, start: float, end: float, steps: int) -> None:
        self.start: float = start
        self.end: float = end
        self.steps: int = max(1, steps)
        self.t: int = 0

    def value(self) -> float:
        """Returns the current epsilon and advances the internal counter."""
        frac = min(1.0, self.t / self.steps)
        eps = self.start + frac * (self.end - self.start)
        self.t += 1
        return eps


class HierarchicalTabular:
    """Two-level hierarchical controller for Chain6.

    Structure:
        - Meta-controller Q2(s, g) selects a sub-goal g ∈ goals.
        - Controller Q1(s, a | g) acts to reach g, with intrinsic rewards.

    Update logic:
        Controller (per-step TD update under a fixed g):
            r_int = 1 if s' == g else 0
            y1 = r_int + gamma1 * max_a' Q1(s', a' | g)
            Q1(s, a | g) ← Q1(s, a | g) + alpha1 * (y1 - Q1(s, a | g))

        Meta-controller (semi-MDP style on option termination):
            Let tau be the number of controller steps taken under g,
            and R_ext the sum of extrinsic rewards observed in that option.
            If terminal:
                y2 = R_ext
            Else:
                y2 = R_ext + (gamma2^tau) * max_g' Q2(s_post, g')
            Q2(s_pre, g) = Q2(s_pre, g) + alpha2 * (y2 - Q2(s_pre, g))

    Attributes:
        goals: Tuple of sub-goal states (e.g., (3, 4, 5, 6)).
        gamma1: Controller discount factor.
        gamma2: Meta-controller discount factor.
        alpha1: Controller learning rate.
        alpha2: Meta-controller learning rate.
        timeout_H: Max controller steps per option (safety cap).
        q_controller: Q1 table with shape (7, 2, |goals|).
        q_meta: Q2 table with shape (7, |goals|).
        goal_to_index: Mapping from sub-goal state to column index.
        epsilon_meta: Epsilon schedule for meta-controller selection.
        epsilon_controller: Epsilon schedules per sub-goal for controller actions.
    """

    def __init__(
        self,
        goals: tuple[int, ...],
        gamma1: float = 0.99,
        gamma2: float = 0.99,
        alpha1: float = 0.2,
        alpha2: float = 0.1,
        timeout_H: int = 10,
    ) -> None:
        self.goals: tuple[int, ...] = goals

        self.gamma1: float = gamma1
        self.gamma2: float = gamma2
        self.alpha1: float = alpha1
        self.alpha2: float = alpha2
        self.timeout_H: int = timeout_H

        # Q1: (state, action, goal_index), Q2: (state, goal_index).
        self.q_controller: np.ndarray = np.zeros((7, 2, len(goals)), dtype=float)
        self.q_meta: np.ndarray = np.zeros((7, len(goals)), dtype=float)

        self.goal_to_index: dict[int, int] = {g: i for i, g in enumerate(goals)}

        # Separate exploration schedules for meta- and controller-levels.
        self.epsilon_meta: EpsilonSchedule = EpsilonSchedule(1.0, 0.1, 5_000)
        self.epsilon_controller: dict[int, EpsilonSchedule] = {
            g: EpsilonSchedule(1.0, 0.1, 1_000) for g in goals
        }

    def select_goal(self, state: int) -> int:
        """Selects a sub-goal via epsilon-greedy over Q2(state, ·)."""
        eps = self.epsilon_meta.value()
        if random.random() < eps:
            return random.choice(self.goals)
        goal_index = int(np.argmax(self.q_meta[state]))
        return self.goals[goal_index]

    def select_action(self, state: int, goal: int) -> int:
        """Selects a primitive action via epsilon-greedy over Q1(state, · | goal)."""
        eps = self.epsilon_controller[goal].value()
        if random.random() < eps:
            return random.choice([PossibleActions.LEFT, PossibleActions.RIGHT]).value
        gi = self.goal_to_index[goal]
        return int(np.argmax(self.q_controller[state, :, gi]))

    def update_controller(
        self,
        state: int,
        action_int: int,
        intrinsic_reward: float,
        next_state: int,
        goal: int,
    ) -> None:
        """Per-step TD update for the controller under the current goal.

        Args:
            state: Current state before taking the action.
            action_int: Executed primitive action (int for PossibleActions).
            intrinsic_reward: Intrinsic reward (1 if next_state == goal else 0).
            next_state: Next environment state.
            goal: Active sub-goal for the controller option.
        """
        gi = self.goal_to_index[goal]
        q_sa = self.q_controller[state, action_int, gi]
        td_target = intrinsic_reward + self.gamma1 * float(
            np.max(self.q_controller[next_state, :, gi])
        )
        self.q_controller[state, action_int, gi] = q_sa + self.alpha1 * (
            td_target - q_sa
        )

    def update_meta(
        self,
        state_pre: int,
        goal: int,
        extrinsic_return: float,
        state_post: int,
        elapsed_steps: int,
        terminal: bool,
    ) -> None:
        """Semi-MDP update for the meta-controller when an option terminates.

        Args:
            state_pre: State where the meta-controller selected `goal`.
            goal: Sub-goal that was pursued by the controller.
            extrinsic_return: Sum of extrinsic rewards observed during this option.
            state_post: State where the option terminated.
            elapsed_steps: Number of controller steps taken under the option.
            terminal: Whether the episode terminated during the option.
        """
        gi = self.goal_to_index[goal]
        q_sg = self.q_meta[state_pre, gi]

        if terminal:
            td_target = extrinsic_return
        else:
            td_target = extrinsic_return + (
                self.gamma2 ** max(1, elapsed_steps)
            ) * float(np.max(self.q_meta[state_post]))

        self.q_meta[state_pre, gi] = q_sg + self.alpha2 * (td_target - q_sg)

    def run_episode(self, env: Chain6Env) -> float:
        """Runs a full episode alternating meta-goal selection and control.

        The meta-controller picks a goal g. The controller then tries to reach g
        for up to `timeout_H` steps, earning intrinsic rewards. Upon option
        termination (hit g or end of episode), the meta-controller is updated
        with the accumulated extrinsic return observed during that option.

        Args:
            env: Instance of Chain6Env.

        Returns:
            Sum of extrinsic rewards collected during the episode.
        """
        state: int = env.reset()
        total_extrinsic_return: float = 0.0

        while True:
            goal: int = self.select_goal(state)

            state_pre: int = state
            option_extrinsic_return: float = 0.0
            elapsed_steps: int = 0
            last_step = None  # for type clarity

            for _ in range(self.timeout_H):
                action_int: int = self.select_action(state, goal)
                last_step = env.step(PossibleActions(action_int))

                intrinsic_reward: float = 1.0 if (last_step.next_state == goal) else 0.0
                self.update_controller(
                    state, action_int, intrinsic_reward, last_step.next_state, goal
                )

                option_extrinsic_return += last_step.reward_ext
                elapsed_steps += 1
                state = last_step.next_state

                if last_step.next_state == goal or last_step.done:
                    break

            # Option terminated; update meta-controller.
            assert last_step is not None  # timeout_H >= 1 guarantees at least one step
            self.update_meta(
                state_pre=state_pre,
                goal=goal,
                extrinsic_return=option_extrinsic_return,
                state_post=state,
                elapsed_steps=elapsed_steps,
                terminal=bool(last_step.done),
            )

            if last_step.done:
                total_extrinsic_return += option_extrinsic_return
                break

        return total_extrinsic_return
