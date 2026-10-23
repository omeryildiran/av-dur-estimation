"""Trial lists with exactly counterbalanced presentation order.

Every staircase and every set of lapse trials gets as many test-first as test-second
trials (an odd count is alternated between staircases, so each condition is balanced).
The whole list is then shuffled, so conditions and staircases are randomly interleaved.
The cross-modal task keeps v1's pre-cued order blocks: each block of
`order_block_size` trials has one order, and blocks alternate.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
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
    """Canonical participant ID: initials ('oy', 'ln2' -> 'OY', 'LN2') or P + number ('p2', '2' -> 'P02').
    SIM IDs are for tests."""
    raw = raw.strip()
    if raw.upper().startswith("SIM"):
        return raw.upper()
    match = re.fullmatch(r"[Pp]?0*(\d+)", raw)
    if match is not None:
        return f"P{int(match.group(1)):02d}"
    if re.fullmatch(r"[A-Za-z]{2,4}\d?", raw):
        return raw.upper()
    raise ValueError(f"Participant ID must be initials (e.g. OY, LN2) or P + number (e.g. P14), got {raw!r}")


def _registry_path(root):
    from avdur import storage
    return Path(root or storage.DATA_ROOT) / "participants.json"


def participant_number(pid, root=None, assign=False):
    """Counterbalancing number. P14 / SIM01 carry it; initials get the next free number in
    <data folder>/participants.json at their first session (assign=True). Before that, the
    number they would get is returned without writing anything."""
    match = re.fullmatch(r"(?:P|SIM)(\d+)", pid)
    if match is not None:
        return int(match.group(1))
    path = _registry_path(root)
    registry = json.loads(path.read_text()) if path.exists() else {}
    if pid in registry:
        return registry[pid]
    number = max(registry.values(), default=0) + 1
    if assign:
        registry[pid] = number
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(registry, indent=2))
    return number


def block_order(pid, root=None, assign=False):
    """Main-experiment block order for this participant (cycles through all 6 orders)."""
    return S.BLOCK_ORDERS[(participant_number(pid, root, assign) - 1) % len(S.BLOCK_ORDERS)]


def crossmodal_first_order(pid, root=None, assign=False):
    """Odd participant numbers start the cross-modal task with A->V, even with V->A."""
    return 1 if participant_number(pid, root, assign) % 2 else 2


def _orders(n, extra_first):
    half = n // 2
    first = half + (n % 2 if extra_first else 0)
    return [1] * first + [2] * (n - first)


def conditions(task, block=None, noise_levels=None, cfg=None):
    """(noise, conflict) pairs; noise_levels overrides the task's (the participant's calibrated high level)."""
    cfg = cfg or S.TASKS[task]
    conflicts = cfg["blocks"][block] if task == "main" else cfg["conflicts"]
    return list(itertools.product(noise_levels or cfg["noise_levels"], conflicts))


def staircase_side(rule):
    """+1 if the rule converges above 50% "test longer" (xD1U, x > 1), -1 below (1DyU), 0 at 50%."""
    n_down, n_up = int(rule[0]), int(rule[2])
    return (n_down > n_up) - (n_down < n_up)


def staircase_starts(task, rng, block=None, noise_levels=None, cfg=None):
    """{(noise, conflict, rule): "floor" | "ceiling"}: every staircase starts on the side opposite
    the point it converges to, so it crosses the PSE on its way (2D1U and 3D1U at the floor,
    1D2U and 1D3U at the ceiling). The staircases come in pairs; a pair of 50% rules (the two
    cross-modal 1D1U) starts at opposite ends, which one at the ceiling random per condition and
    balanced over the session's conditions."""
    rules = (cfg or S.TASKS[task])["staircases"]
    if len(rules) % 2:
        raise ValueError(f"{task}: staircases must come in pairs, got {rules}")
    conds = conditions(task, block, noise_levels, cfg)
    starts = {}
    for k in range(0, len(rules), 2):
        a, b = rules[k], rules[k + 1]
        side_a, side_b = staircase_side(a), staircase_side(b)
        if side_a and side_b:
            if side_a == side_b:
                raise ValueError(f"{task}: a pair must converge on opposite sides of 50%, got {a}, {b}")
            for noise, conflict in conds:
                starts[(noise, conflict, a)] = "floor" if side_a > 0 else "ceiling"
                starts[(noise, conflict, b)] = "floor" if side_b > 0 else "ceiling"
            continue
        if side_a or side_b:
            raise ValueError(f"{task}: pair {a}, {b} mixes a 50% rule with another")
        n = len(conds)
        first_on_top = [True] * (n // 2) + [False] * (n // 2) + ([bool(rng.random() < 0.5)] if n % 2 else [])
        rng.shuffle(first_on_top)
        for (noise, conflict), top in zip(conds, first_on_top):
            starts[(noise, conflict, a)] = "ceiling" if top else "floor"
            starts[(noise, conflict, b)] = "floor" if top else "ceiling"
    return starts


def build_trials(task, rng, block=None, practice=False, first_order=1, noise_levels=None, cfg=None):
    cfg = cfg or S.TASKS[task]
    slots = []
    if practice:
        conds = conditions(task, block, noise_levels, cfg)
        for i in range(S.PRACTICE_TRIALS):
            noise, conflict = conds[i % len(conds)]
            level = cfg["lapse_levels"][i % 2]
            slots.append(TrialSpec(task, noise, conflict, "practice", None, level, 1 + i % 2))
    else:
        for noise, conflict in conditions(task, block, noise_levels, cfg):
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
