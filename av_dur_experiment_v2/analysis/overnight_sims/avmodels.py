"""Deterministic (quadrature) likelihood of the three bimodal observer models.

Same models as monteCarloClass (lognorm = causal inference, fusionOnlyLogNorm, switchingFree):
log-space Gaussian measurements, boxcar prior on [ln t_min, ln t_max], estimates are posterior
means of a truncated normal, CI averages the common-cause (fused) and auditory-only estimates by
the posterior of a common cause. The answer is "test longer" when the test's estimate exceeds
the standard's; a lapse gives a random answer.

Instead of Monte Carlo draws, the two measurements of an interval are integrated on a fixed
grid (K x K points over +-5 sigma), the standard's estimate is turned into a CDF, and
P(test longer) = sum over the test's grid of w * F_standard(estimate_test). Deterministic, so
BADS needs no noise handling, and about 100x faster than nSimul 1000.

Parameters (as monteCarloClass with sharedLambda, freeP_c False):
  ci:     [lapse, sa_low, sv, p_c, sa_high]
  fusion: [lapse, sa_low, sv, sa_high]
  switch: [lapse, sa_low, sv, p_switch_low, sa_high, p_switch_high]

Audiovisual bias (optional, bias=(c_low, c_high, d)): the visual measurement's mean is
(1 - d) ln v + d ln 0.5 - c[noise] instead of ln v, in both intervals (ln v_hat = ln v - c - d ln(v / 0.5)).
c > 0: vision seems shorter; d > 0: the bias grows with duration.

v1 data (Data(..., vis_offset={noise_idx: k})): every visual flash lasted k s longer than nominal (the
cross-modal correction), so the visual durations are the nominal ones + k; the auditory ones are nominal.
"""

import numpy as np
from scipy.special import ndtr

SQ2PI = np.sqrt(2 * np.pi)
K = 33                                   # 2-D grid (causal inference); fusion and switching need 1-D only
_Z = np.linspace(-5, 5, K)
_W1 = np.exp(-_Z ** 2 / 2); _W1 /= _W1.sum()
Z1, Z2 = (a.ravel() for a in np.meshgrid(_Z, _Z, indexing="ij"))
W2 = np.outer(_W1, _W1).ravel()
K1 = 201
_Z1D = np.linspace(-6, 6, K1)
_W1D = np.exp(-_Z1D ** 2 / 2); _W1D /= _W1D.sum()

T_MIN, T_MAX = 0.025, 0.95          # boxcar prior = the test range (floor -95%, ceiling +90% of 500 ms)
LO, HI = np.log(T_MIN), np.log(T_MAX)
NPARAMS = {"ci": 5, "fusion": 4, "switch": 6}


def _phi(x):
    return np.exp(-x * x / 2) / SQ2PI


def tn_mean(mu, s):
    a, b = (LO - mu) / s, (HI - mu) / s
    Z = ndtr(b) - ndtr(a)
    ok = Z > 1e-300
    est = mu + s * (_phi(a) - _phi(b)) / np.where(ok, Z, 1.0)
    est = np.where(ok & np.isfinite(est), est, np.clip(mu, LO, HI))
    return np.clip(est, LO, HI)


def unpack(model, p, noise_idx):
    """noise_idx 0 = low, 1 = high -> (lapse, sa, sv, pc_or_pswitch)."""
    if model == "ci":
        lam, sa1, sv, pc, sa2 = p
        return lam, (sa1, sa2)[noise_idx], sv, pc
    if model == "fusion":
        lam, sa1, sv, sa2 = p
        return lam, (sa1, sa2)[noise_idx], sv, 1.0
    lam, sa1, sv, ps1, sa2, ps2 = p
    return lam, (sa1, sa2)[noise_idx], sv, (ps1, ps2)[noise_idx]


def estimates(model, la, lv, sa, sv, pc):
    """Estimates (log) and weights over the measurement grid for true log durations la, lv
    (arrays broadcast against the grid: shape (..., 1)). Returns (E, W) with E (..., n), W (n,)."""
    if model == "switch":        # each estimate depends on one measurement: 1-D grids
        ea = tn_mean(la + sa * _Z1D, sa); ev = tn_mean(lv + sv * _Z1D, sv)
        return np.concatenate([ea, ev], -1), np.r_[(1 - pc) * _W1D, pc * _W1D]
    ja, jv = 1 / sa ** 2, 1 / sv ** 2
    sc = np.sqrt(1 / (ja + jv))
    if model == "fusion" or pc >= 1:    # the fused mean is Gaussian: 1-D grid
        wa = ja / (ja + jv)
        sd = np.sqrt((wa * sa) ** 2 + ((1 - wa) * sv) ** 2)
        return tn_mean(wa * la + (1 - wa) * lv + sd * _Z1D, sc), _W1D
    ma = la + sa * Z1
    mv = lv + sv * Z2
    muc = (ja * ma + jv * mv) / (ja + jv)
    e1 = tn_mean(muc, sc)
    width = HI - LO
    L1 = (1 / width) * sc / (SQ2PI * sa * sv) * (ndtr((HI - muc) / sc) - ndtr((LO - muc) / sc)) \
        * np.exp(-(ma - mv) ** 2 / (2 * (sa ** 2 + sv ** 2)))
    pa = (ndtr((HI - ma) / sa) - ndtr((LO - ma) / sa)) / width
    pv = (ndtr((HI - mv) / sv) - ndtr((LO - mv) / sv)) / width
    num = pc * L1; den = num + (1 - pc) * pa * pv
    post = np.where(den > 0, num / np.where(den > 0, den, 1), pc)
    e2 = tn_mean(ma, sa)
    return post * e1 + (1 - post) * e2, W2


def _cdf(E, W):
    o = np.argsort(E, kind="stable"); es = E[o]; cw = np.cumsum(W[o]) - W[o] / 2
    return es, cw


LSTD = np.log(0.5)


def vis(lv, bias, noise_idx):
    """Mean of the visual measurement for true log duration lv."""
    if bias is None:
        return lv
    c, d = bias[noise_idx], bias[2]
    return lv - c - d * (lv - LSTD)


def p_longer(model, p, noise_idx, std_a, std_v, tests, bias=None):
    """P(test longer) for one condition: standard (std_a, std_v) s, test durations `tests` (s, AV equal)."""
    lam, sa, sv, pc = unpack(model, p, noise_idx)
    es, cw = _cdf(*estimates(model, np.log(std_a), vis(np.log(std_v), bias, noise_idx), sa, sv, pc))
    lt = np.log(np.asarray(tests, float))[:, None]
    Et, Wt = estimates(model, lt, vis(lt, bias, noise_idx), sa, sv, pc)
    return (1 - lam) * (np.interp(Et, es, cw, left=0.0, right=1.0) @ Wt) + lam / 2


class Data:
    """Aggregated trials: per noise level the union of tests; per condition the indices into it, n and k."""

    def __init__(self, noise_idx, conflict, test, chose, std=0.5, vis_offset=None):
        self.std = std
        self.vis_offset = dict(vis_offset or {})
        test = np.round(np.asarray(test, float), 5)
        self.tests, self.cells = {}, []
        for ni in np.unique(noise_idx):
            self.tests[int(ni)] = np.unique(test[noise_idx == ni])
        keys = np.c_[noise_idx, np.round(conflict, 4)]
        for ni, c in np.unique(keys, axis=0):
            m = (keys[:, 0] == ni) & (keys[:, 1] == c)
            ut, inv = np.unique(test[m], return_inverse=True)
            k = np.bincount(inv, weights=np.asarray(chose, float)[m]); n = np.bincount(inv).astype(float)
            self.cells.append((int(ni), float(c), np.searchsorted(self.tests[int(ni)], ut), n, k))
        self.n_trials = len(test)


def nll(model, p, data, bias=None):
    tot = 0.0
    for ni, tests in data.tests.items():
        lam, sa, sv, pc = unpack(model, p, ni)
        off = getattr(data, "vis_offset", {}).get(ni, 0.0)
        lt = np.log(tests)[:, None]
        lvt = np.log(np.maximum(tests + off, 0.01))[:, None] if off else lt
        Et, Wt = estimates(model, lt, vis(lvt, bias, ni), sa, sv, pc)
        for cni, c, idx, n, k in data.cells:
            if cni != ni:
                continue
            es, cw = _cdf(*estimates(model, np.log(data.std), vis(np.log(data.std + c + off), bias, ni), sa, sv, pc))
            pr = (1 - lam) * (np.interp(Et[idx], es, cw, left=0.0, right=1.0) @ Wt) + lam / 2
            pr = np.clip(pr, 1e-9, 1 - 1e-9)
            tot -= np.sum(k * np.log(pr) + (n - k) * np.log1p(-pr))
    return tot
