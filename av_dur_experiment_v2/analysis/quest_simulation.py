"""QUEST+ (psi-marginal) vs the current staircases for one main-experiment condition.

    python analysis/quest_simulation.py            # full run (about 10 min on 8 cores)
    python analysis/quest_simulation.py --quick    # small run to check the code

The question: how many trials does a Bayesian adaptive method need to estimate the PSE (mu)
and sigma of one condition as precisely as the current design (2D1U + 1D2U, 70 trials each,
plus 14 easy trials = 154 trials per condition)?

Both methods are simulated on the same observers and analysed with the same maximum-likelihood
fit as the experiment (psychometrics.fit: lognormal, P(test longer) = lapse/2 + (1 - lapse) *
Phi((ln(test/standard) - mu) / sigma), lapse 0-0.1). Each simulated run is 154 trials long and is
evaluated after its first 40, 60, 80, 100, 120 and 154 trials (stopping a run early gives the
same trials as a shorter run, for both methods). The staircases are the experiment's own
avdur.staircase.Staircase with the main-task settings.

Outputs go to analysis/quest_simulation_results/ (CSV + figures).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import logistic, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiment_settings as S  # noqa: E402
from avdur.staircase import Staircase  # noqa: E402

OUT = ROOT / "analysis" / "quest_simulation_results"
MAIN = S.TASKS["main"]
N_PER_STAIR = MAIN["trials_per_staircase"]                   # 70
N_EASY = 2 * MAIN["lapse_trials_per_level"]                  # 14
N_TOTAL = 2 * N_PER_STAIR + N_EASY                           # 154
BUDGETS = (40, 60, 80, 100, 120, N_TOTAL)
FLOOR, CEIL = MAIN["floor"], S.STAIRCASE["ceiling"]          # -0.95, 0.9 (test/standard - 1)
EASY_X = (np.log(1 + FLOOR), np.log(1 + CEIL))               # easy trials: 25 ms and 950 ms vs 500 ms
LAPSE_MAX = S.BIAS["lapse_max"]                              # 0.1, as the block fits


# ── QUEST+ (psi-marginal) ──────────────────────────────────────────────────────
class QuestPlus:
    """Bayesian adaptive method over (mu, sigma, lapse) for P(test longer).

    Before each trial it picks the test duration whose answer is expected to leave the
    posterior over (mu, sigma) with the least entropy; the lapse is a nuisance parameter
    (psi-marginal: Prins 2013), so trials are not spent on measuring it. After the answer the
    posterior is updated with Bayes' rule. Kontsevich & Tyler 1999 (psi), Watson 2017 (QUEST+).
    """

    def __init__(self, x_grid=None, mu_grid=None, sigma_grid=None, lapse_grid=None, top_k=1, target="mu_sigma"):
        self.x = np.linspace(-1.6, np.log(1 + CEIL), 57) if x_grid is None else np.asarray(x_grid)
        self.mu = np.linspace(-1.0, 1.0, 81) if mu_grid is None else np.asarray(mu_grid)
        self.sigma = np.exp(np.linspace(np.log(0.05), np.log(2.5), 25)) if sigma_grid is None else np.asarray(sigma_grid)
        self.lapse = np.linspace(0, LAPSE_MAX, 5) if lapse_grid is None else np.asarray(lapse_grid)
        m, s, l = np.meshgrid(self.mu, self.sigma, self.lapse, indexing="ij")
        self.p_mu, self.p_sigma, self.p_lapse = m.ravel(), s.ravel(), l.ravel()
        # likelihood table: P(test longer | x, theta), shape (n_x, n_theta)
        self.table = self.p_lapse / 2 + (1 - self.p_lapse) * norm.cdf(
            (self.x[:, None] - self.p_mu[None, :]) / self.p_sigma[None, :])
        self.post = np.full(self.p_mu.size, 1.0 / self.p_mu.size)    # flat in mu, ln sigma and lapse
        self.n_ms = self.mu.size * self.sigma.size
        self.top_k = top_k
        self.target = target            # "mu_sigma": PSE and slope (standard psi-marginal); "mu": PSE only

    def _entropy_ms(self, joint):
        """Entropy of the target marginal ((mu, sigma) or mu) for each row of an unnormalised joint posterior."""
        if self.target == "mu":
            ms = joint.reshape(joint.shape[0], self.mu.size, -1).sum(-1)
        else:
            ms = joint.reshape(joint.shape[0], self.n_ms, self.lapse.size).sum(-1)
        ms = ms / ms.sum(1, keepdims=True)
        return -np.sum(ms * np.log(np.where(ms > 0, ms, 1.0)), axis=1)

    def next_x(self, rng=None):
        p1 = self.table @ self.post                                   # P(test longer) per x
        h1 = self._entropy_ms(self.table * self.post)
        h0 = self._entropy_ms((1 - self.table) * self.post)
        expected = p1 * h1 + (1 - p1) * h0
        if self.top_k > 1 and rng is not None:                        # optional variety: one of the best k
            return float(self.x[rng.choice(np.argsort(expected)[:self.top_k])])
        return float(self.x[np.argmin(expected)])

    def update(self, x, chose_test):
        p = self.p_lapse / 2 + (1 - self.p_lapse) * norm.cdf((x - self.p_mu) / self.p_sigma)
        self.post = self.post * (p if chose_test else 1 - p)
        self.post /= self.post.sum()

    def estimate(self):
        """Posterior means (mu, sigma via ln sigma, lapse)."""
        return (float(self.post @ self.p_mu), float(np.exp(self.post @ np.log(self.p_sigma))),
                float(self.post @ self.p_lapse))


# ── observers ──────────────────────────────────────────────────────────────────
class Observer:
    """P(test judged longer). Options: logistic shape, order bias (+beta when the test is second),
    linear drift of mu and sigma over the run (learning)."""

    def __init__(self, mu, sigma, lapse, shape="normal", order_bias=0.0, drift_mu=0.0, sigma_end=None, n=N_TOTAL):
        self.mu, self.sigma, self.lapse = mu, sigma, lapse
        self.shape, self.order_bias, self.drift_mu = shape, order_bias, drift_mu
        self.sigma_end = sigma if sigma_end is None else sigma_end
        self.n = n

    def p(self, x, trial, order):
        f = trial / max(self.n - 1, 1)
        mu = self.mu + self.drift_mu * f
        sigma = self.sigma * (self.sigma_end / self.sigma) ** f
        z = (x - mu + (self.order_bias if order == 2 else -self.order_bias)) / sigma
        core = logistic.cdf(z * 1.7) if self.shape == "logistic" else norm.cdf(z)
        return self.lapse / 2 + (1 - self.lapse) * core

    def answer(self, x, trial, order, rng):
        return bool(rng.random() < self.p(x, trial, order))


# ── one run of each method ─────────────────────────────────────────────────────
def _schedule(rng):
    """Trial types in random order: 'a' (2D1U), 'b' (1D2U) / 'q' (QUEST+), 'e' easy. Same for both."""
    kinds = np.array(["s"] * (2 * N_PER_STAIR) + ["e"] * N_EASY)
    rng.shuffle(kinds)
    easy = iter(np.r_[[EASY_X[0]] * (N_EASY // 2), [EASY_X[1]] * (N_EASY // 2)][rng.permutation(N_EASY)])
    orders = rng.integers(1, 3, N_TOTAL)
    return kinds, easy, orders


def run_staircases(obs, rng):
    kinds, easy, orders = _schedule(rng)
    stairs = {"2D1U": Staircase("2D1U", FLOOR, CEIL, start="floor"), "1D2U": Staircase("1D2U", FLOOR, CEIL, start="ceiling")}
    which = iter(np.r_[["2D1U"] * N_PER_STAIR, ["1D2U"] * N_PER_STAIR][rng.permutation(2 * N_PER_STAIR)])
    xs, ys = [], []
    for t, k in enumerate(kinds):
        if k == "e":
            x = next(easy); y = obs.answer(x, t, orders[t], rng)
        else:
            st = stairs[next(which)]
            x = np.log(1 + st.level); y = obs.answer(x, t, orders[t], rng); st.update(y)
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys, float), None


def run_quest(obs, rng, quest_kwargs=None, record_posterior_at=()):
    kinds, easy, orders = _schedule(rng)
    q = QuestPlus(**(quest_kwargs or {}))
    xs, ys, post_est = [], [], {}
    for t, k in enumerate(kinds):
        x = next(easy) if k == "e" else q.next_x(rng)
        y = obs.answer(x, t, orders[t], rng)
        q.update(x, y)                          # easy trials inform the posterior too (mostly the lapse)
        xs.append(x); ys.append(y)
        if t + 1 in record_posterior_at:
            post_est[t + 1] = q.estimate()
    return np.array(xs), np.array(ys, float), post_est


# ── the experiment's fit (fast version of psychometrics.fit for one condition) ─
_MU_G = np.linspace(-1.2, 1.2, 49)
_SIG_G = np.exp(np.linspace(np.log(0.03), np.log(2.9), 30))
_LAM_G = np.array([0.0, 0.05, LAPSE_MAX])


def fit_ml(x, y, lapse_fixed=None):
    """ML fit with the experiment's model and bounds (mu +/-1.5, sigma 0.01-3, lapse 0-0.1):
    coarse grid, then L-BFGS-B from the best grid point. lapse_fixed: lapse known (as when one
    lapse is shared over a participant's conditions)."""
    ux, inv = np.unique(np.round(x, 6), return_inverse=True)
    k = np.bincount(inv, weights=y); n = np.bincount(inv).astype(float)
    lams = _LAM_G if lapse_fixed is None else np.array([lapse_fixed])
    pr = lams[None, None, :, None] / 2 + (1 - lams[None, None, :, None]) * norm.cdf(
        (ux[None, None, None, :] - _MU_G[:, None, None, None]) / _SIG_G[None, :, None, None])
    pr = np.clip(pr, 1e-9, 1 - 1e-9)
    ll = (k * np.log(pr) + (n - k) * np.log(1 - pr)).sum(-1)
    i, j, l = np.unravel_index(np.argmax(ll), ll.shape)

    def nll(p):
        lam = lapse_fixed if lapse_fixed is not None else p[2]
        q = np.clip(lam / 2 + (1 - lam) * norm.cdf((ux - p[0]) / p[1]), 1e-9, 1 - 1e-9)
        return -np.sum(k * np.log(q) + (n - k) * np.log(1 - q))

    x0 = [_MU_G[i], _SIG_G[j]] + ([] if lapse_fixed is not None else [lams[l]])
    bounds = [(-1.5, 1.5), (0.01, 3.0)] + ([] if lapse_fixed is not None else [(0.0, LAPSE_MAX)])
    r = minimize(nll, x0, bounds=bounds, method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1])


# ── one simulated observer, both methods, every budget ─────────────────────────
def simulate_one(args):
    scenario, rep, obs_kw, seed, quest_kwargs, methods = args
    rows = []
    for method in methods:
        rng = np.random.default_rng([seed, 0 if method == "staircase" else 1])
        obs = Observer(**obs_kw)
        if method == "staircase":
            xs, ys, post = run_staircases(obs, rng)
        else:
            xs, ys, post = run_quest(obs, rng, quest_kwargs, record_posterior_at=BUDGETS)
        for b in BUDGETS:
            mu_f, sig_f = fit_ml(xs[:b], ys[:b])
            mu_k, sig_k = fit_ml(xs[:b], ys[:b], lapse_fixed=obs.lapse if obs.lapse <= LAPSE_MAX else LAPSE_MAX)
            row = dict(scenario=scenario, rep=rep, method=method, budget=b, true_mu=obs_kw["mu"],
                       true_sigma=obs_kw["sigma"], true_lapse=obs_kw["lapse"], mu_hat=mu_f, sigma_hat=sig_f,
                       mu_hat_lapse_known=mu_k, sigma_hat_lapse_known=sig_k,
                       share_near_pse=float(np.mean(np.abs(xs[:b] - obs_kw["mu"]) < obs_kw["sigma"])))
            if post:
                row.update(mu_post=post[b][0], sigma_post=post[b][1])
            row["method"] = method if method == "staircase" or not quest_kwargs else "quest_" + quest_kwargs["target"]
            rows.append(row)
    return rows


def scenarios(n_rep, rng, v1_fits=None):
    jobs = []

    def add(name, n, make):
        for r in range(n):
            jobs.append((name, r, make(r), int(rng.integers(2**31)), None, ("staircase", "quest")))

    for sigma in (0.2, 0.35, 0.6, 0.9):                           # main grid
        for lapse in (0.0, 0.05, 0.1):
            add(f"grid s{sigma} l{lapse}", n_rep,
                lambda r, s=sigma, l=lapse: dict(mu=float(rng.uniform(-0.35, 0.35)), sigma=s, lapse=l))
    if v1_fits is not None:                                       # v1 participants' conditions
        ok = v1_fits[(v1_fits.sigma > 0.03) & (v1_fits.sigma < 2.9)].reset_index(drop=True)
        add("v1 realistic", 3 * n_rep, lambda r: (lambda p: dict(mu=float(p.mu), sigma=float(p.sigma),
                                                                lapse=float(p.lapse)))(ok.iloc[rng.integers(len(ok))]))
    base = dict(sigma=0.5, lapse=0.05)                            # robustness, around a typical condition
    add("robust: baseline", n_rep, lambda r: dict(mu=float(rng.uniform(-0.35, 0.35)), **base))
    add("robust: lapse 0.15", n_rep, lambda r: dict(mu=float(rng.uniform(-0.35, 0.35)), sigma=0.5, lapse=0.15))
    add("robust: learning", n_rep, lambda r: dict(mu=float(rng.uniform(-0.35, 0.35)), sigma=0.65, sigma_end=0.35,
                                                  drift_mu=0.1, lapse=0.05))
    add("robust: order bias", n_rep, lambda r: dict(mu=float(rng.uniform(-0.35, 0.35)), order_bias=0.25, **base))
    add("robust: logistic shape", n_rep, lambda r: dict(mu=float(rng.uniform(-0.35, 0.35)), shape="logistic", **base))
    add("robust: extreme PSE", n_rep, lambda r: dict(mu=float(rng.choice([-0.75, 0.55])), **base))
    return jobs


def summarise(df):
    out = []
    for (sc, m, b), g in df.groupby(["scenario", "method", "budget"]):
        e_mu = g.mu_hat - g.true_mu
        e_ms = 500 * (np.exp(g.mu_hat) - np.exp(g.true_mu))          # PSE error in ms at the 500 ms standard
        e_ls = np.log(g.sigma_hat / g.true_sigma)
        e_mu_k = g.mu_hat_lapse_known - g.true_mu
        pse_mae = float(np.median(np.abs(e_ms)))
        e_ls_k = np.log(g.sigma_hat_lapse_known / g.true_sigma)
        out.append(dict(scenario=sc, method=m, budget=b, n=len(g),
                        pse_rmse_ms=float(np.sqrt(np.mean(e_ms ** 2))), pse_mae_ms=pse_mae, pse_bias_ms=float(np.mean(e_ms)),
                        mu_rmse=float(np.sqrt(np.mean(e_mu ** 2))),
                        sigma_rmse_ln=float(np.sqrt(np.mean(e_ls ** 2))), sigma_median_ratio=float(np.exp(np.median(e_ls))),
                        sigma_fail=float(np.mean((g.sigma_hat > 2.9) | (g.sigma_hat < 0.02))),
                        mu_rmse_lapse_known=float(np.sqrt(np.mean(e_mu_k ** 2))),
                        sigma_rmse_ln_lapse_known=float(np.sqrt(np.mean(e_ls_k ** 2))),
                        share_near_pse=float(g.share_near_pse.mean())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--quest-target", choices=("mu_sigma", "mu"),
                    help="run only QUEST+ with this target (results saved with that suffix)")
    args = ap.parse_args()
    n_rep = 6 if args.quick else args.reps
    OUT.mkdir(parents=True, exist_ok=True)
    v1 = pd.read_csv(OUT / "v1_main_fits.csv") if (OUT / "v1_main_fits.csv").exists() else None
    jobs = scenarios(n_rep, np.random.default_rng(2026), v1)
    tag = "_quick" if args.quick else ""
    if args.quest_target:                      # same observers and seeds, QUEST+ variant only
        jobs = [j[:4] + ({"target": args.quest_target}, ("quest",)) for j in jobs]
        tag += "_" + args.quest_target
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [r for chunk in pool.imap_unordered(simulate_one, jobs, chunksize=4) for r in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"raw{tag}.csv", index=False)
    summ = summarise(df)
    summ.to_csv(OUT / f"summary{tag}.csv", index=False)
    print(f"{len(jobs)} observers x 2 methods in {time.time() - t0:.0f} s")
    print(summ[summ.budget.isin([80, N_TOTAL])].round(3).to_string())


if __name__ == "__main__":
    main()
