"""Observer models of the 2IFC duration task, with P("test longer") computed exactly or by quadrature
(no Monte Carlo, so the likelihood is smooth and the same at every evaluation).

Generative model (v1 article, lognormal versions; everything in log seconds):
    each interval gives an auditory measurement m_A ~ N(log S_A, sigma_A) and, in the main experiment, a
    visual one m_V ~ N(log S_V, sigma_V); sigma_A depends on the noise level. sigma_* are SINGLE-INTERVAL
    noises: a unimodal psychometric sigma is sqrt(2) times larger.
    The observer's estimate of each interval uses a boxcar prior on log duration, [log T_MIN, log T_MAX]
    (v1: the range of test durations), so single-cue and fused estimates are truncated-normal means
    (tn_mean). The answer is "test longer" if est(test) > est(standard); with probability lapse the answer
    is a coin flip: P = lapse/2 + (1 - lapse) * P(est_test > est_standard).
    Standard: S_A = 0.5 s, S_V = 0.5 s + conflict. Test: S_A = S_V = test duration.

Models (v1 names in brackets):
    fusion     forced fusion [fusionOnlyLogNorm]: est = tn_mean of the reliability-weighted average.
               tn_mean is increasing and sigma_c is the same in both intervals, so
               P = Phi((log t - w_A log S_A - w_V log S_V) / (sqrt 2 sigma_c))           (exact)
    switching  cue switching [switchingFree]: each interval independently uses the visual estimate with
               probability p_switch (per noise level), else the auditory one, each tn_mean of its own cue.
               Same-cue pairs are exact Phi terms; mixed pairs are a 1-D Gauss-Hermite integral.
    causal     causal inference, model averaging [lognorm]: est = pi * tn_mean(fused) +
               (1 - pi) * tn_mean(m_A), pi = posterior of a common cause (prior p_c) under the boxcar prior
               (Koerding et al. 2007 with a uniform prior). est is increasing in m_A for fixed m_V (checked in
               verify.py), so P(est_t > e) given m_V,t is 1 - Phi((h(e) - log t) / sigma_A), h the inverse in
               m_A; the rest is Gauss-Hermite quadrature over m_V,t and the standard's (m_A, m_V).
Unimodal blocks: P = Phi((log t - log 0.5) / (sqrt 2 sigma)) for every model (tn_mean is increasing).

simulate() draws the measurements and applies the same estimates (Monte Carlo); verify.py checks the
exact/quadrature probabilities against it and against v1's monteCarloClass.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, log_ndtr, ndtr

from . import data as D

T_MIN, T_MAX = 0.05, 2.0            # boxcar prior: the shortest and longest test durations of the v2 design
BOX = (float(np.log(T_MIN)), float(np.log(T_MAX)))
LOG_STD = float(np.log(D.STANDARD_S))
SQRT2 = np.sqrt(2.0)
N_GH = 24                           # Gauss-Hermite nodes per dimension (verify.py: < 1e-3 from 1e6 MC draws)
N_GRID = 401                        # m_A grid for inverting the causal-inference estimate
N_E = 1500                          # grid of estimates (log s, over the box) for the causal-inference CDFs
MODELS = ("fusion", "causal", "switching")

_GH_X, _GH_W = np.polynomial.hermite.hermgauss(N_GH)
GH_X, GH_W = SQRT2 * _GH_X, _GH_W / np.sqrt(np.pi)      # nodes and weights for a standard normal


def _phi(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)


def _log_box_mass(m, s, box):
    """log(Phi((b - m)/s) - Phi((a - m)/s)), accurate far outside the box."""
    hi, lo = (box[1] - m) / s, (box[0] - m) / s
    upper = lo > 0                                   # both in the upper tail: use the mirrored form
    p, q = np.where(upper, -lo, hi), np.where(upper, -hi, lo)
    lp, lq = log_ndtr(p), log_ndtr(q)
    return lp + np.log1p(-np.exp(np.minimum(lq - lp, -1e-300)))


def _tn_and_mass(m, sigma, box=BOX):
    """(tn_mean, log box mass) of N(m, sigma) on box, sharing the log-CDF evaluations."""
    a, b = box
    m = np.asarray(m, float)
    al, be = (a - m) / sigma, (b - m) / sigma
    log_mass = _log_box_mass(m, sigma, box)
    log_z = log_mass + 0.5 * np.log(2 * np.pi)
    ratio = np.exp(-0.5 * al * al - log_z) - np.exp(-0.5 * be * be - log_z)     # (phi(al) - phi(be)) / Z
    est = m + sigma * ratio
    return np.clip(np.where(np.isfinite(est), est, np.clip(m, a, b)), a, b), log_mass


def tn_mean(m, sigma, box=BOX):
    """Posterior mean of log duration: likelihood N(m, sigma), boxcar prior on box (v1 truncated_normal_mean,
    with the normalising mass in log space so it stays accurate far outside the box)."""
    return _tn_and_mass(m, sigma, box)[0]


def weights(sa, sv):
    w_a = sv ** 2 / (sa ** 2 + sv ** 2)
    return w_a, 1 - w_a, sa * sv / np.sqrt(sa ** 2 + sv ** 2)


def posterior_common(ma, mv, sa, sv, pc, box=BOX):
    """P(common cause | m_A, m_V) under the boxcar prior (v1 posterior_C1, computed in log space so it
    does not fall back to the prior where v1's likelihoods underflow)."""
    w_a, _, sc = weights(sa, sv)
    mc = w_a * ma + (1 - w_a) * mv
    width = box[1] - box[0]
    # N(m_A; y, sa) N(m_V; y, sv) = N(m_A - m_V; 0, sqrt(sa^2 + sv^2)) N(y; mc, sc), integrated over the box
    log_l1 = (np.log(sc / (np.sqrt(2 * np.pi) * sa * sv)) - (ma - mv) ** 2 / (2 * (sa ** 2 + sv ** 2))
              + _log_box_mass(mc, sc, box) - np.log(width))
    log_l2 = _log_box_mass(ma, sa, box) + _log_box_mass(mv, sv, box) - 2 * np.log(width)
    pc = np.clip(pc, 1e-12, 1 - 1e-12)
    return expit(np.log(pc) + log_l1 - np.log1p(-pc) - log_l2)


def est_causal(ma, mv, sa, sv, pc, box=BOX):
    w_a, _, sc = weights(sa, sv)
    post = posterior_common(ma, mv, sa, sv, pc, box)
    return post * tn_mean(w_a * ma + (1 - w_a) * mv, sc, box) + (1 - post) * tn_mean(ma, sa, box)


def est_causal_grid(ma, mv, sa, sv, pc, box=BOX):
    """est_causal on the grid ma (G,) x mv (K,) -> (K, G); the auditory-only and visual-only terms are
    computed once per axis (same values as est_causal)."""
    w_a, w_v, sc = weights(sa, sv)
    width = box[1] - box[0]
    tn_a, lm_a = _tn_and_mass(ma, sa, box)                                  # (G,)
    lm_v = _log_box_mass(mv, sv, box)                                       # (K,)
    tn_c, lm_c = _tn_and_mass(w_a * ma[None, :] + w_v * mv[:, None], sc, box)
    log_l1 = (np.log(sc / (np.sqrt(2 * np.pi) * sa * sv))
              - (ma[None, :] - mv[:, None]) ** 2 / (2 * (sa ** 2 + sv ** 2)) + lm_c - np.log(width))
    log_l2 = lm_a[None, :] + lm_v[:, None] - 2 * np.log(width)
    pc = np.clip(pc, 1e-12, 1 - 1e-12)
    post = expit(np.log(pc) + log_l1 - np.log1p(-pc) - log_l2)
    return post * tn_c + (1 - post) * tn_a[None, :]


# ── P(test longer) without lapse ─────────────────────────────────────────────

def p_unimodal(log_t, sigma, log_s=LOG_STD):
    return ndtr((np.asarray(log_t) - log_s) / (SQRT2 * sigma))


def p_fusion(log_t, log_sv, sa, sv, log_sa=LOG_STD):
    w_a, w_v, sc = weights(sa, sv)
    return ndtr((np.asarray(log_t) - w_a * log_sa - w_v * log_sv) / (SQRT2 * sc))


def _inverse_tn(sigma, box=BOX, n=N_GRID):
    """Grid (m, tn_mean(m)) for inverting tn_mean with np.interp."""
    m = np.linspace(box[0] - 8 * sigma - 1, box[1] + 8 * sigma + 1, n)
    return tn_mean(m, sigma, box), m


def p_mixed(mu_x, sx, mu_y, sy, box=BOX):
    """P(tn_mean(m_x; sx) > tn_mean(m_y; sy)), m_x ~ N(mu_x, sx), m_y ~ N(mu_y, sy). mu_x, mu_y broadcast."""
    mu_x, mu_y = np.broadcast_arrays(np.asarray(mu_x, float), np.asarray(mu_y, float))
    ey = tn_mean(mu_y[..., None] + sy * GH_X, sy, box)               # (..., N_GH)
    g, m = _inverse_tn(sx, box)
    thr = np.interp(ey, g, m)                                        # m_x giving the same estimate
    return np.sum(GH_W * (1 - ndtr((thr - mu_x[..., None]) / sx)), axis=-1)


def p_switching(log_t, log_sv, sa, sv, p_v, log_sa=LOG_STD, box=BOX):
    log_t = np.asarray(log_t, float)
    aa = ndtr((log_t - log_sa) / (SQRT2 * sa))
    vv = ndtr((log_t - log_sv) / (SQRT2 * sv))
    av = p_mixed(log_t, sa, log_sv, sv, box)          # test auditory, standard visual
    va = p_mixed(log_t, sv, log_sa, sa, box)          # test visual, standard auditory
    q = 1 - p_v
    return q * q * aa + p_v * p_v * vv + q * p_v * av + p_v * q * va


class CausalNoiseLevel:
    """P(test longer) under causal inference for one noise level (sigma_A, sigma_V, p_c fixed).

    The two intervals' estimates are independent, so P(est_test > est_standard) = integral of
    (1 - F_test) dF_standard, with F the CDF of an interval's estimate. Given m_V the estimate increases with
    m_A, so F(e | m_V) = Phi((h(e; m_V) - mu_A) / sigma_A), h the inverse in m_A (interpolated on a grid);
    F(e) averages that over m_V by Gauss-Hermite. The CDFs are built once per test level and once per
    conflict, then every (test, conflict) pair is one trapezoid sum over the estimate grid."""

    def __init__(self, sa, sv, pc, box=BOX, n_gh=N_GH, n_grid=N_GRID, n_e=N_E):
        self.sa, self.sv, self.pc, self.box = sa, sv, pc, box
        self.grid = np.linspace(box[0] - 6 * sa - 0.5, box[1] + 6 * sa + 0.5, n_grid)
        self.e = np.linspace(box[0], box[1], n_e)
        x, w = np.polynomial.hermite.hermgauss(n_gh)
        self.x, self.w = SQRT2 * x, w / np.sqrt(np.pi)

    def cdf(self, mu_a, mu_v):
        """CDF of one interval's estimate on self.e (measurement means mu_a, mu_v)."""
        mv = mu_v + self.sv * self.x                                        # (K,)
        est = est_causal_grid(self.grid, mv, self.sa, self.sv, self.pc, self.box)
        est = np.maximum.accumulate(est, axis=1)                            # guard tiny numerical dips
        h = np.stack([np.interp(self.e, row, self.grid) for row in est])    # (K, n_e): m_A giving estimate e
        f = self.w @ ndtr((h - mu_a) / self.sa)
        f[-1] = 1.0                                                         # every estimate lies in the box
        return f

    def p(self, log_ts, log_svs):
        """P(est_test > est_standard) for pairs (test level, visual standard), arrays of equal length."""
        log_ts, log_svs = np.asarray(log_ts, float), np.asarray(log_svs, float)
        ts, t_idx = np.unique(np.round(log_ts, 10), return_inverse=True)
        svs, s_idx = np.unique(np.round(log_svs, 10), return_inverse=True)
        surv = np.stack([1 - self.cdf(t, t) for t in ts])                   # test: S_A = S_V = t
        surv = 0.5 * (surv[:, 1:] + surv[:, :-1])
        dstd = np.diff(np.stack([self.cdf(LOG_STD, s) for s in svs]), axis=1)
        return np.einsum("ij,ij->i", surv[t_idx], dstd[s_idx])


# ── Monte Carlo (simulation and checks) ──────────────────────────────────────

def estimates(model, ma, mv, sa, sv, extra, rng, box=BOX):
    """Estimates of a batch of intervals (log s) from measurements. extra: p_c (causal) or p_switch."""
    if model == "fusion":
        w_a, _, sc = weights(sa, sv)
        return tn_mean(w_a * ma + (1 - w_a) * mv, sc, box)
    if model == "causal":
        return est_causal(ma, mv, sa, sv, extra, box)
    if model == "switching":
        use_v = rng.random(np.shape(ma)) < extra
        return np.where(use_v, tn_mean(mv, sv, box), tn_mean(ma, sa, box))
    raise ValueError(model)


def simulate_p(model, log_t, log_sv, sa, sv, extra, n=200_000, rng=None, log_sa=LOG_STD, box=BOX):
    """Monte Carlo P(est_test > est_standard) for one stimulus (no lapse)."""
    rng = rng or np.random.default_rng(0)
    ma_s, mv_s = rng.normal(log_sa, sa, n), rng.normal(log_sv, sv, n)
    ma_t, mv_t = rng.normal(log_t, sa, n), rng.normal(log_t, sv, n)
    es = estimates(model, ma_s, mv_s, sa, sv, extra, rng, box)
    et = estimates(model, ma_t, mv_t, sa, sv, extra, rng, box)
    return float(np.mean(et > es))
