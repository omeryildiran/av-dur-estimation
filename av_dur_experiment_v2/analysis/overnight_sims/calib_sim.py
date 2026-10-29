"""Auditory noise calibration: the current staircases vs QUEST+ over (noise level, test duration).

    python analysis/overnight_sims/calib_sim.py --n 300 --workers 1

Goal of the calibration: the noise level n* where the participant's auditory sigma is 0.75.

Simulated listeners (P101-P104 calibrations: the time-order error is large, accuracy 58-70% with
the test first vs 83-87% with it second):
  sigma(n) = sqrt(s0^2 + (k n)^(2p)), s0 U(0.15, 0.5), p U(0.6, 2.2) (shallow to steep, P103's fitted
  slope was 2.1), k such that sigma(n*) = 0.75 with n* log-uniform 0.3-1.9;
  P(test judged longer) = lapse/2 + (1 - lapse) Phi((ln(test/std) - mu + s beta) / sigma(n)),
  s = +1 when the test is second; mu N(0, 0.1), beta U(-0.1, 0.4), lapse U(0, 0.08).

Methods (all with 20 easy catch trials at low noise, the new minimum):
  stair  the current calibration (avdur.noise_calibration): two 2D1U staircases on ln(noise), test
         fixed at +-0.41, estimate() as in the experiment; 60 or 100 trials per staircase
  q_n    QUEST+ choosing noise level and test duration (order balanced), model ln sigma = ln 0.75 +
         b (ln n - ln n*), with mu, beta (order bias) and lapse as nuisance; target n* only
  q_nb   the same, targeting n* and b
Scored by the sigma the participant actually has at the chosen level: ln(sigma(n_hat) / 0.75).
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import experiment_settings as S  # noqa: E402
from avdur import noise_calibration as NC  # noqa: E402
from avdur.staircase import Staircase  # noqa: E402

C = S.NOISE_CALIBRATION
TARGET = C["target_sigma"]
LO, HI = C["bounds"]
N_CATCH = 20
STD = 0.5


class Listener:
    def __init__(self, rng):
        self.n_star = float(np.exp(rng.uniform(np.log(0.3), np.log(1.9))))
        self.s0, self.p = rng.uniform(0.15, 0.5), rng.uniform(0.6, 2.2)
        self.k = (TARGET ** 2 - self.s0 ** 2) ** (1 / (2 * self.p)) / self.n_star
        self.mu, self.beta, self.lapse = rng.normal(0, 0.1), rng.uniform(-0.1, 0.4), rng.uniform(0, 0.08)

    def sigma(self, n):
        return np.sqrt(self.s0 ** 2 + (self.k * np.asarray(n)) ** (2 * self.p))

    def answer(self, x, n, order, rng):
        s = 1.0 if order == 2 else -1.0
        p = self.lapse / 2 + (1 - self.lapse) * norm.cdf((x - self.mu + s * self.beta) / self.sigma(n))
        return bool(rng.random() < p)


class QuestCalib:
    """QUEST+ over (ln n*, b, mu, beta, lapse); stimuli (noise, ln test/std); the order is given per trial."""

    def __init__(self, target="n"):
        self.ln_ns = np.linspace(np.log(0.08), np.log(3.0), 36)
        self.b = np.exp(np.linspace(np.log(0.25), np.log(2.5), 8))
        self.mu = np.linspace(-0.3, 0.3, 5)
        self.beta = np.linspace(-0.3, 0.6, 7)
        self.lam = np.array([0.0, 0.04, 0.08])
        g = np.meshgrid(self.ln_ns, self.b, self.mu, self.beta, self.lam, indexing="ij")
        self.th = [a.ravel().astype(np.float32) for a in g]
        self.shape = tuple(len(v) for v in (self.ln_ns, self.b, self.mu, self.beta, self.lam))
        self.noise = np.exp(np.linspace(np.log(LO), np.log(HI), 15))
        mags = np.geomspace(0.05, 1.2, 8)
        self.x = np.r_[-mags[::-1], mags]
        nn, xx = np.meshgrid(self.noise, self.x, indexing="ij")
        self.stim = np.c_[nn.ravel(), xx.ravel()]
        self.tables = {o: self._p(self.stim[:, 0], self.stim[:, 1], o) for o in (1, 2)}
        self.post = np.full(self.th[0].size, 1.0 / self.th[0].size, np.float32)
        self.target = target

    def _p(self, n, x, order):
        ln_ns, b, mu, beta, lam = self.th
        s = 1.0 if order == 2 else -1.0
        sig = np.exp(np.log(TARGET) + b[None, :] * (np.log(np.atleast_1d(n))[:, None] - ln_ns[None, :]))
        z = (np.atleast_1d(x)[:, None] - mu[None, :] + s * beta[None, :]) / sig
        return (lam / 2 + (1 - lam) * norm.cdf(z)).astype(np.float32)

    def _entropy(self, joint):
        j = joint.reshape((joint.shape[0],) + self.shape)
        m = j.sum(axis=(2, 3, 4, 5)) if self.target == "n" else j.sum(axis=(3, 4, 5)).reshape(j.shape[0], -1)
        m = m / m.sum(1, keepdims=True)
        return -np.sum(m * np.log(np.where(m > 0, m, 1.0)), axis=1)

    def next(self, order):
        t = self.tables[order]
        p1 = t @ self.post
        h = p1 * self._entropy(t * self.post) + (1 - p1) * self._entropy((1 - t) * self.post)
        return tuple(self.stim[int(np.argmin(h))])

    def update(self, n, x, order, y):
        p = self._p(n, x, order)[0]
        self.post *= p if y else 1 - p
        self.post /= self.post.sum()

    def estimate(self):
        return float(np.exp(np.clip(self.post @ self.th[0], np.log(LO), np.log(HI))))


def run_stair(L, n_per, rng):
    levels = NC.LEVELS
    stairs = {}
    for name, start in zip(("A", "B"), C["start_noise"]):
        st = Staircase("2D1U", floor=-np.log(HI), ceiling=-np.log(LO), init_step=C["init_step_ln"],
                       step_factor=C["step_factor"], n_step_changes=C["n_step_changes"])
        st.level = float(-np.log(start)); stairs[name] = st
    slots = [("staircase", nm, (o, lv)) for nm in stairs for k, (o, lv) in
             zip(range(n_per), [(o, lv) for o in (1, 2) for lv in levels] * n_per)]
    slots += [("lapse", None, (1 + (k // 2) % 2, (-0.9, 0.9)[k % 2])) for k in range(N_CATCH)]
    rng.shuffle(slots)
    rows = []
    for i, (kind, nm, (order, lv)) in enumerate(slots):
        if kind == "staircase":
            n = float(np.exp(-stairs[nm].level)); x = float(np.log(1 + lv))
        else:
            n = S.NOISE_LOW; x = float(np.log(1 + lv))
        y = L.answer(x, n, order, rng)
        correct = y == (x > 0)
        rev = False
        if kind == "staircase":
            stairs[nm].update(correct); rev = stairs[nm].is_reversal
        rows.append(dict(kind=kind, trial_num=i, current_stair=nm, stair_is_reversal=rev, audNoise=n,
                         testDurS=STD * np.exp(x), standardDur=STD, is_correct=float(correct)))
    return NC.estimate(pd.DataFrame(rows))["noise_high"]


def run_quest(L, n_total, target, rng):
    q = QuestCalib(target)
    kinds = np.r_[["q"] * (n_total - N_CATCH), ["e"] * N_CATCH]
    rng.shuffle(kinds)
    orders = rng.permutation(np.r_[[1, 2] * (len(kinds) // 2 + 1)][:len(kinds)])
    out, n_seen = {}, 0
    for k, order in zip(kinds, orders):
        if k == "e":
            n, x = S.NOISE_LOW, float(np.log(1 + (0.9 if rng.random() < 0.5 else -0.9)))
        else:
            n, x = q.next(order)
        q.update(n, x, order, L.answer(x, n, order, rng))
        n_seen += 1
        if n_seen in (100, 140, 200) and n_seen <= n_total:
            out[n_seen] = q.estimate()
    return out


def one(args):
    i, seed = args
    rng = np.random.default_rng(seed)
    L = Listener(rng)
    rows = []
    for n_per in (60, 100):
        nh = run_stair(L, n_per, np.random.default_rng([seed, n_per]))
        rows.append(dict(method="stair", trials=2 * n_per + N_CATCH, n_hat=nh))
    for target in ("n", "nb"):
        for tr, nh in run_quest(L, 200, target, np.random.default_rng([seed, 7, len(target)])).items():
            rows.append(dict(method=f"q_{target}", trials=tr, n_hat=nh))
    for r in rows:
        r.update(i=i, n_star=L.n_star, p=L.p, s0=L.s0, beta=L.beta, mu=L.mu, lapse=L.lapse,
                 sigma_err=float(np.log(L.sigma(r["n_hat"]) / TARGET)), n_err=float(np.log(r["n_hat"] / L.n_star)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    seeds = np.random.default_rng(2610).integers(2**31, size=args.n)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [r for chunk in pool.imap_unordered(one, list(enumerate(seeds))) for r in chunk]
    df = pd.DataFrame(rows)
    out = HERE / "results" / "calib_sim.csv"
    df.to_csv(out, index=False)
    g = df.groupby(["method", "trials"])
    print(pd.DataFrame(dict(sigma_rmse=g.sigma_err.apply(lambda e: np.sqrt(np.mean(e ** 2))),
                            sigma_med_abs=g.sigma_err.apply(lambda e: np.median(np.abs(e))),
                            off_by_x1_5=g.sigma_err.apply(lambda e: np.mean(np.abs(e) > np.log(1.5))),
                            sigma_bias=g.sigma_err.mean())).round(3))
    print(f"{args.n} listeners in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
