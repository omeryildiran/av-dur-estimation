"""Trial lists with exactly counterbalanced presentation order.

Every staircase and every set of lapse trials gets as many test-first as test-second
trials (an odd count is alternated between staircases, so each condition is balanced).
The whole list is then shuffled, so conditions and staircases are randomly interleaved.
The cross-modal task keeps v1's pre-cued order blocks: each block of
`order_block_size` trials has one order, and blocks alternate.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, asdict

import experiment_settings as S


@dataclass
class TrialSpec:
    task: str
    noise: float | None
    conflict: float
    kind: str                      # "staircase", "lapse" or "practice"
    staircase: str | None          # rule of the staircase that sets the test level
    fixed_level: float | None      # level for lapse/practice trials ("floor" resolved later)
    order: int                     # 1 = test first, 2 = test second
    order_block: int | None = None # cross-modal mini-block index

    @property
    def condition(self):
        return (self.noise, self.conflict)

    def to_dict(self):
        return asdict(self)


def normalize_pid(raw):
    """Canonical participant ID: 'p2', '2', 'P2' and 'P02' all become 'P02'. SIM IDs are for tests."""
    raw = raw.strip()
    if raw.upper().startswith("SIM"):
        return raw.upper()
    match = re.fullmatch(r"[Pp]?0*(\d+)", raw)
    if match is None:
        raise ValueError(f"Participant ID must be P followed by a number (e.g. P14), got {raw!r}")
    return f"P{int(match.group(1)):02d}"


def participant_number(pid):
    match = re.search(r"(\d+)$", pid)
    if match is None:
        raise ValueError(f"Participant ID must end in a number (e.g. P14), got {pid!r}")
    return int(match.group(1))


def block_order(pid):
    """Main-experiment block order for this participant (cycles through all 6 orders)."""
    return S.BLOCK_ORDERS[(participant_number(pid) - 1) % len(S.BLOCK_ORDERS)]


def crossmodal_first_order(pid):
    """Odd participant numbers start the cross-modal task with A->V, even with V->A."""
    return 1 if participant_number(pid) % 2 else 2


def _orders(n, extra_first):
    half = n // 2
    first = half + (n % 2 if extra_first else 0)
    return [1] * first + [2] * (n - first)


def conditions(task, block=None):
    cfg = S.TASKS[task]
    conflicts = cfg["blocks"][block] if task == "main" else cfg["conflicts"]
    return list(itertools.product(cfg["noise_levels"], conflicts))


def staircase_starts(task, rng, block=None):
    """{(noise, conflict, rule): "floor" | "ceiling"}: the staircases are taken in pairs, and in
    each condition one of each pair starts at each end. Which one starts at the ceiling is
    random per condition, with each rule at the ceiling in half of the conditions."""
    rules = S.TASKS[task]["staircases"]
    if len(rules) % 2:
        raise ValueError(f"{task}: staircases must come in pairs, got {rules}")
    conds = conditions(task, block)
    starts = {}
    for k in range(0, len(rules), 2):
        n = len(conds)
        first_on_top = [True] * (n // 2) + [False] * (n // 2) + ([bool(rng.random() < 0.5)] if n % 2 else [])
        rng.shuffle(first_on_top)
        for (noise, conflict), top in zip(conds, first_on_top):
            starts[(noise, conflict, rules[k])] = "ceiling" if top else "floor"
            starts[(noise, conflict, rules[k + 1])] = "floor" if top else "ceiling"
    return starts


def build_trials(task, rng, block=None, practice=False, first_order=1):
    cfg = S.TASKS[task]
    slots = []
    if practice:
        conds = conditions(task, block)
        for i in range(S.PRACTICE_TRIALS):
            noise, conflict = conds[i % len(conds)]
            level = cfg["lapse_levels"][i % 2]
            slots.append(TrialSpec(task, noise, conflict, "practice", None, level, 1 + i % 2))
    else:
        for noise, conflict in conditions(task, block):
            for k, rule in enumerate(cfg["staircases"]):
                for order in _orders(cfg["trials_per_staircase"], extra_first=(k % 2 == 0)):
                    slots.append(TrialSpec(task, noise, conflict, "staircase", rule, None, order))
            for k, level in enumerate(cfg["lapse_levels"]):
                for order in _orders(cfg["lapse_trials_per_level"], extra_first=(k % 2 == 0)):
                    slots.append(TrialSpec(task, noise, conflict, "lapse", None, level, order))

    rng.shuffle(slots)
    if task != "crossmodal":
        return slots

    # cross-modal: fill alternating single-order blocks from the two order pools
    size = cfg["order_block_size"]
    pools = {1: [s for s in slots if s.order == 1], 2: [s for s in slots if s.order == 2]}
    trials, order, block_index = [], first_order, 0
    while pools[1] or pools[2]:
        if not pools[order]:
            order = 3 - order
        chunk, pools[order] = pools[order][:size], pools[order][size:]
        for s in chunk:
            s.order_block = block_index
        trials.extend(chunk)
        order, block_index = 3 - order, block_index + 1
    return trials
