"""Transformed up/down staircase with the v1 step rules.

A rule "xDyU" means: after x consecutive "test longer" answers the test gets shorter
(down), after y consecutive "standard longer" answers it gets longer (up). Every rule of
this form converges; v1's "1U1D" (which moved away from threshold) cannot be expressed.
A suffix after an underscore only names a second staircase with the same rule ("1D1U_b").
`start` is "floor" or "ceiling": v2 starts one staircase of each pair at each end
(design.staircase_starts); v1 started every staircase at the floor.
"""

from __future__ import annotations

import re


class Staircase:
    def __init__(self, rule, floor, ceiling=0.9, init_step=0.16, step_factor=0.5, n_step_changes=2, start="floor"):
        match = re.fullmatch(r"(\d)D(\d)U(_[a-z0-9]+)?", rule)
        if match is None:
            raise ValueError(f"Staircase rule must look like '2D1U' or '1D1U_b', got {rule!r}")
        if start not in ("floor", "ceiling"):
            raise ValueError(f"start must be 'floor' or 'ceiling', got {start!r}")
        self.rule = rule
        self.n_down, self.n_up = int(match.group(1)), int(match.group(2))
        self.floor, self.ceiling = float(floor), float(ceiling)
        self.init_step, self.step_factor, self.n_step_changes = init_step, step_factor, n_step_changes
        self.start = start
        self.level = self.ceiling if start == "ceiling" else self.floor
        self.step = init_step
        self.reversals = 0
        self.is_reversal = False
        self._test_streak = 0
        self._standard_streak = 0
        self._directions = []
        self.n_trials = 0

    def update(self, chose_test):
        """Record one answer (True = test judged longer) and move the level if a streak completes."""
        self.n_trials += 1
        self.is_reversal = False
        if chose_test:
            self._test_streak += 1
            self._standard_streak = 0
            if self._test_streak == self.n_down:
                self._test_streak = 0
                self._move(-1)
        else:
            self._standard_streak += 1
            self._test_streak = 0
            if self._standard_streak == self.n_up:
                self._standard_streak = 0
                self._move(+1)

    def _move(self, direction):
        self._directions.append(direction)
        if len(self._directions) >= 2 and self._directions[-1] != self._directions[-2]:
            self.is_reversal = True
            self.reversals += 1
            self.step = self.init_step * self.step_factor ** min(self.reversals, self.n_step_changes)
        self.level = min(max(self.level + direction * self.step, self.floor), self.ceiling)
