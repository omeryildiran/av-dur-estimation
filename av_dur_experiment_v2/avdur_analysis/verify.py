"""Double checks of the models (run_all() returns one row per check; the modelling notebook shows them).

    v1_pieces      tn_mean, posterior of a common cause and the causal-inference estimate against v1's
                   monteCarloClass (repo root), on random measurements
    against_mc     P(test longer) of every model (exact / quadrature) against Monte Carlo with 10^6 draws:
                   our simulate_p, and v1's probTestLonger_vectorized_mc
    monotone       the causal-inference estimate increases with m_A at fixed m_V (the quadrature needs it)
    sqrt2          2IFC: a psychometric fit to simulated unimodal answers returns sqrt(2) x single-interval sigma
    pse_log        forced fusion: the PSE is 0.5 s^w_A x (0.5 s + conflict)^w_V (combination in log time),
                   not w_A 0.5 + w_V (0.5 + conflict) (linear time)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import data as D
from . import models as M

REPO = D.ROOT.parent
CASES = [  # (sigma_A, sigma_V, p_c or p_switch, conflict s, test s)
    (0.10, 0.30, 0.7, 0.25, 0.55), (0.45, 0.30, 0.5, -0.45, 0.40), (0.80, 0.25, 0.9, 0.45, 0.80),
    (0.25, 0.60, 0.3, -0.1667, 0.30), (1.20, 0.35, 0.6, 0.0833, 1.20), (0.35, 0.35, 0.8, 0.45, 0.50),
]


def _v1():
    """v1's OmerMonteCarlo with only the attributes its model methods use."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from monteCarloClass import OmerMonteCarlo
    obj = object.__new__(OmerMonteCarlo)
    obj.nSimul = 1_000_000
    return obj


def v1_pieces(seed=0):
    rng = np.random.default_rng(seed)
    v1 = _v1()
    a, b = M.BOX
    rows = []
    for sa, sv, pc, _, _ in CASES:
        ma, mv = rng.normal(-0.7, 1.0, 2000), rng.normal(-0.7, 1.0, 2000)
        # v1 underflows (posterior falls back to p_c, truncated mean to the box edge) for measurements far
        # outside the box; ours is computed in log space there. Compare where v1's arithmetic holds.
        near = (ma > a - 5 * sa) & (ma < b + 5 * sa) & (mv > a - 5 * sv) & (mv < b + 5 * sv)
        checks = {
            "tn_mean": (M.tn_mean(ma, sa), v1.truncated_normal_mean(ma, sa, a, b)),
            "posterior C=1": (M.posterior_common(ma, mv, sa, sv, pc), v1.posterior_C1(ma, mv, sa, sv, pc, a, b)),
            "causal estimate": (M.est_causal(ma, mv, sa, sv, pc), v1.causalInference_vectorized(ma, mv, sa, sv, pc, a, b)),
        }
        for name, (ours, theirs) in checks.items():
            rows.append(dict(check=f"v1 {name}", case=f"sA={sa} sV={sv} p={pc}; {int((~near).sum())} far draws left out",
                             max_abs_diff=float(np.max(np.abs(ours - theirs)[near]))))
    return pd.DataFrame(rows)


def _ours(model, sa, sv, extra, c, t):
    lt, lsv = np.log(t), np.log(D.STANDARD_S + c)
    if model == "fusion":
        return float(M.p_fusion(lt, lsv, sa, sv))
    if model == "switching":
        return float(M.p_switching(lt, lsv, sa, sv, extra))
    return float(M.CausalNoiseLevel(sa, sv, extra).p([lt], [lsv])[0])


V1_NAMES = {"fusion": "fusionOnlyLogNorm", "causal": "lognorm", "switching": "switchingFree"}


def against_mc(n=1_000_000, seed=1, with_v1=True):
    v1 = _v1() if with_v1 else None
    rows = []
    for model in M.MODELS:
        for sa, sv, extra, c, t in CASES:
            ours = _ours(model, sa, sv, extra, c, t)
            mc = M.simulate_p(model, np.log(t), np.log(D.STANDARD_S + c), sa, sv, extra, n=n,
                              rng=np.random.default_rng(seed))
            se = np.sqrt(mc * (1 - mc) / n) * (2 if model == "switching" else 1.5)
            row = dict(check=f"P(test longer) {model}", case=f"sA={sa} sV={sv} p={extra} c={c:+.3f} t={t}",
                       exact=ours, mc=mc, max_abs_diff=abs(ours - mc), mc_se=se)
            if v1 is not None:
                np.random.seed(seed)
                v1.modelName = V1_NAMES[model]
                stims = (D.STANDARD_S, t, D.STANDARD_S + c, t)
                row["v1_mc"] = float(v1.probTestLonger_vectorized_mc(stims, sa, sv, extra, 0.0, M.T_MIN, M.T_MAX))
                row["diff_v1"] = abs(ours - row["v1_mc"])
            rows.append(row)
    return pd.DataFrame(rows)


def monotone(n_cases=200, seed=2):
    rng = np.random.default_rng(seed)
    worst = 0.0
    ma = np.linspace(M.BOX[0] - 3, M.BOX[1] + 3, 2001)
    for _ in range(n_cases):
        sa, sv, pc = rng.uniform(0.05, 1.5), rng.uniform(0.05, 1.5), rng.uniform(0.01, 0.99)
        mv = rng.uniform(M.BOX[0] - 1, M.BOX[1] + 1)
        e = M.est_causal(ma, mv, sa, sv, pc)
        worst = min(worst, float(np.diff(e).min()))
    return pd.DataFrame([dict(check="causal estimate increasing in m_A", case=f"{n_cases} random (sA, sV, pc, mV)",
                              max_abs_diff=-worst)])


def sqrt2(seed=3, n_per_level=400):
    from . import psychometric as P
    rng = np.random.default_rng(seed)
    rows = []
    for s in (0.1, 0.3, 0.8):
        x = np.linspace(-3 * s, 3 * s, 15)
        k = rng.binomial(n_per_level, M.p_unimodal(x + M.LOG_STD, s))
        d = pd.DataFrame({"condition": "u", "x": np.repeat(x, n_per_level),
                          "chose_test": np.concatenate([np.r_[np.ones(kk), np.zeros(n_per_level - kk)] for kk in k])})
        f = P.fit(d)
        rows.append(dict(check="2IFC: psychometric sigma / sigma_single", case=f"sigma_single={s}",
                         exact=np.sqrt(2), mc=float(f.sigma[0] / s), max_abs_diff=abs(f.sigma[0] / s - np.sqrt(2))))
    return pd.DataFrame(rows)


def pse_log():
    rows = []
    for sa, sv, _, c, _ in CASES:
        w_a, w_v, _ = M.weights(sa, sv)
        pse = brentq(lambda t: M.p_fusion(np.log(t), np.log(D.STANDARD_S + c), sa, sv) - 0.5, 0.01, 5)
        log_rule = D.STANDARD_S ** w_a * (D.STANDARD_S + c) ** w_v
        linear_rule = w_a * D.STANDARD_S + w_v * (D.STANDARD_S + c)
        rows.append(dict(check="fusion PSE = log-time weighting", case=f"sA={sa} sV={sv} c={c:+.3f}",
                         exact=log_rule, mc=pse, max_abs_diff=abs(pse - log_rule),
                         linear_rule_ms_off=1000 * (linear_rule - pse)))
    return pd.DataFrame(rows)


def run_all(with_v1=True, n_mc=1_000_000):
    parts = [monotone(), sqrt2(), pse_log(), against_mc(n_mc, with_v1=with_v1)]
    if with_v1:
        parts.insert(0, v1_pieces())
    return pd.concat(parts, ignore_index=True)
