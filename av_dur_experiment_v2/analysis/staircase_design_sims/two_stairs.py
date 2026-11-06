"""4x35 vs two 70-trial staircases, unimodal block, one condition (140 staircase + 20 easy)."""
import sys, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd
V2 = Path("/Users/ofy204/Documents/my_repos/av-dur-estimation/av_dur_experiment_v2")
sys.path[:0] = [str(V2), str(V2 / "analysis")]
from scipy.stats import norm
from scipy.optimize import minimize
class _Obs:
    def __init__(self, mu, sigma, lapse, n=None): self.mu, self.sigma, self.lapse = mu, sigma, lapse
    def answer(self, x, t, order, rng):
        p = self.lapse / 2 + (1 - self.lapse) * norm.cdf((x - self.mu) / self.sigma)
        return int(rng.random() < p)
_MU = np.linspace(-1.5, 1.5, 61); _SG = np.exp(np.linspace(np.log(0.02), np.log(3), 40)); _LM = np.array([0, .025, .05, .075, .1])
def _fit(x, y):
    ux, inv = np.unique(np.round(x, 6), return_inverse=True)
    k = np.bincount(inv, weights=y); n = np.bincount(inv).astype(float)
    pr = np.clip(_LM[None,None,:,None]/2 + (1-_LM[None,None,:,None])*norm.cdf((ux[None,None,None,:]-_MU[:,None,None,None])/_SG[None,:,None,None]), 1e-9, 1-1e-9)
    ll = (k*np.log(pr) + (n-k)*np.log(1-pr)).sum(-1); i, j, l = np.unravel_index(np.argmax(ll), ll.shape)
    def nll(p):
        q = np.clip(p[2]/2 + (1-p[2])*norm.cdf((ux-p[0])/p[1]), 1e-9, 1-1e-9); return -np.sum(k*np.log(q)+(n-k)*np.log(1-q))
    r = minimize(nll, [_MU[i], _SG[j], _LM[l]], bounds=[(-1.5,1.5),(0.01,3.0),(0,0.1)], method="L-BFGS-B"); return r.x[0], r.x[1]
class qs:
    norm_cdf = staticmethod(norm.cdf)
    Observer = _Obs; fit_ml = staticmethod(_fit)
from avdur.staircase import Staircase
import experiment_settings as S

DESIGNS = {
    "4x35 (now)": ((("2D1U", "floor"), ("1D2U", "ceiling"), ("3D1U", "floor"), ("1D3U", "ceiling")), 35),
    "2D1U+1D3U x70": ((("2D1U", "floor"), ("1D3U", "ceiling")), 70),
    "3D1U+1D2U x70": ((("3D1U", "floor"), ("1D2U", "ceiling")), 70),
    "2D1U+1D2U x70": ((("2D1U", "floor"), ("1D2U", "ceiling")), 70),
}
N_EASY = 20
TASK = {"auditory": S.TASKS["unimodal_auditory"], "visual": S.TASKS["unimodal_visual"]}

def run(design, obs, rng, task):
    rules, n = DESIGNS[design]
    cfg = TASK[task]
    k = len(rules)
    kinds = np.array(["s"] * (k * n) + ["e"] * N_EASY); rng.shuffle(kinds)
    lo, hi = cfg["lapse_levels"]
    easy = iter(np.log(1 + np.r_[[lo] * 10, [hi] * 10][rng.permutation(N_EASY)]))
    orders = rng.integers(1, 3, len(kinds))
    st = [Staircase(r, cfg["floor"], S.STAIRCASE["ceiling"], init_step=S.STAIRCASE["init_step"],
                    step_factor=S.STAIRCASE["step_factor"], n_step_changes=S.STAIRCASE["n_step_changes"],
                    start=s, start_top=S.STAIRCASE["start_top"]) for r, s in rules]
    which = iter(np.repeat(np.arange(k), n)[rng.permutation(k * n)])
    xs, ys = [], []
    for t, kd in enumerate(kinds):
        if kd == "e":
            x = next(easy); y = obs.answer(x, t, orders[t], rng)
        else:
            s = st[next(which)]; x = np.log(1 + s.level); y = obs.answer(x, t, orders[t], rng); s.update(y)
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys, float)

def one(a):
    task, sig, lapse, rep, seed = a
    mu = float(np.random.default_rng(seed).uniform(-0.15, 0.15))
    out = []
    for d in DESIGNS:
        rng = np.random.default_rng([seed, list(DESIGNS).index(d)])
        obs = qs.Observer(mu=mu, sigma=sig, lapse=lapse, n=160)
        xs, ys = run(d, obs, rng, task)
        m, s = qs.fit_ml(xs, ys)
        out.append(dict(task=task, sigma=sig, lapse=lapse, design=d, err_ln_sigma=np.log(s / sig), err_pse_ms=500 * (np.exp(m) - np.exp(mu))))
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    js = [(t, s, l, r, int(rng.integers(2**31))) for t, ss in (("auditory", (0.3, 0.6, 1.2)), ("visual", (0.6, 1.0)))
          for s in ss for l in (0.0, 0.05) for r in range(300)]
    with mp.get_context("spawn").Pool(max(1, mp.cpu_count() - 1)) as p:
        rows = [x for c in p.imap_unordered(one, js, chunksize=8) for x in c]
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parent / "unimodal_sim.csv", index=False)
    g = df.groupby(["task", "sigma", "design"], sort=False)
    res = pd.DataFrame({"sigma_rmse_%": 100 * np.sqrt(g.err_ln_sigma.apply(lambda e: np.mean(e**2))),
                        "sigma_bias_%": 100 * g.err_ln_sigma.mean(),
                        "pse_rmse_ms": np.sqrt(g.err_pse_ms.apply(lambda e: np.mean(e**2)))}).round(1)
    print(res.to_string())
