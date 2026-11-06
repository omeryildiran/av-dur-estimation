"""Main experiment, one noise level, 9 conflicts, 3 parts (part 1 on day 1, parts 2-3 on day 2).
The listener may change between days. Designs (per condition):
  continue      2D1U + 1D3U, 17 trials per part, staircases carry on (current)
  continue+step as continue, but each part goes back to the 80 ms step (halved at that part's first 2 reversals)
  restart       fresh staircases each part (2D1U from the floor, 1D3U from 950 ms)
  MCS narrow    constant stimuli, 7 fixed test durations per condition, log-spaced 0.55-1.8x standard (275-900 ms)
  MCS wide      the same over 0.3-3.3x (150-1650 ms)
Staircase designs: 102 staircase trials + 96 easy per noise level (~113 per condition). MCS: 7 x 17 = 119 per
condition, no extra easy trials. Fit per condition over all parts (lognormal, lapse <= 0.1).
Truth for the pooled fit: trial-weighted mean PSE (1/3 day 1, 2/3 day 2); sigma moment-matched to the mixture."""
import sys, multiprocessing as mp
import numpy as np, pandas as pd
from scipy.stats import norm
SP = str(__import__("pathlib").Path(__file__).resolve().parent)
sys.path[:0] = ["/Users/ofy204/Documents/my_repos/av-dur-estimation/av_dur_experiment_v2", SP]
from two_stairs import qs
from avdur.staircase import Staircase
import experiment_settings as S
CONF = np.array(S.TASKS["main"]["blocks"]["part1"]); NC = len(CONF); FLOOR = S.TASKS["main"]["floor"]
PER_PART = 17; N_EASY_PART = 32; LAPSE = 0.03
DESIGNS = ["continue", "continue+step", "restart", "MCS narrow", "MCS wide"]
MCS_LEVELS = {"MCS narrow": np.exp(np.linspace(np.log(0.55), np.log(1.8), 7)) - 1,
              "MCS wide": np.exp(np.linspace(np.log(0.3), np.log(3.3), 7)) - 1}

class Stair(Staircase):
    rev0 = 0
    def reset_step(self):
        self.step, self.rev0 = self.init_step, self.reversals
    def _move(self, d):
        self._directions.append(d)
        if len(self._directions) >= 2 and self._directions[-1] != self._directions[-2]:
            self.is_reversal = True; self.reversals += 1
            self.step = self.init_step * self.step_factor ** min(self.reversals - self.rev0, self.n_step_changes)
        self.level = min(max(self.level + d * self.step, self.floor), self.ceiling)

def mk(rule, start):
    return Stair(rule, FLOOR, S.STAIRCASE["ceiling"], init_step=0.16, step_factor=0.5, n_step_changes=2, start=start, start_top=0.9)
RULES = (("2D1U", "floor"), ("1D3U", "ceiling"))

def answer(x, mu, sig, rng):
    return int(rng.random() < LAPSE / 2 + (1 - LAPSE) * norm.cdf((x - mu) / sig))

def run(design, params, rng):
    """params[part] = (mus per condition, sigma). Returns per-condition x, y."""
    X = [[] for _ in range(NC)]; Y = [[] for _ in range(NC)]
    stairs = None
    for part in range(3):
        mus, sig = params[part]
        if design.startswith("MCS"):
            lv = MCS_LEVELS[design]
            trials = [(c, np.log(1 + l)) for c in range(NC) for l in lv for _ in range(17 // 3 + (1 if part < 17 % 3 else 0))]
            for i in rng.permutation(len(trials)):
                c, x = trials[i]; X[c].append(x); Y[c].append(answer(x, mus[c], sig, rng))
            continue
        if stairs is None or design == "restart":
            stairs = {(c, k): mk(*RULES[k]) for c in range(NC) for k in range(2)}
        elif design == "continue+step":
            for st in stairs.values(): st.reset_step()
        trials = [("s", c, k) for c in range(NC) for k in range(2) for _ in range(PER_PART)]
        trials += [("e", int(rng.integers(NC)), i % 2) for i in range(N_EASY_PART)]
        for i in rng.permutation(len(trials)):
            kind, c, k = trials[i]
            if kind == "e":
                x = np.log(1 + (FLOOR if k == 0 else S.TEST_CEILING)); y = answer(x, mus[c], sig, rng)
            else:
                st = stairs[(c, k)]; x = np.log(1 + st.level); y = answer(x, mus[c], sig, rng); st.update(y)
            X[c].append(x); Y[c].append(y)
    return X, Y

SCEN = {"no change": (0.0, 1.0), "PSE +80 ms on day 2": (0.15, 1.0), "sigma x1.5 on day 2": (0.0, 1.5), "sigma x0.67 on day 2": (0.0, 0.67)}

def one(a):
    sig, w, scen, seed = a
    shift, sf = SCEN[scen]
    base = np.log(1 + w * CONF / 0.5) + np.random.default_rng(seed).uniform(-0.1, 0.1)
    s2 = sig * sf
    params = [(base, sig), (base + shift, s2), (base + shift, s2)]
    true_mu = base + 2 / 3 * shift
    true_sig = np.sqrt(sig**2 / 3 + 2 / 3 * s2**2 + (1 / 3) * (2 / 3) * shift**2)
    out = []
    for i, d in enumerate(DESIGNS):
        X, Y = run(d, params, np.random.default_rng([seed, i]))
        for c in range(NC):
            m, s = qs.fit_ml(np.array(X[c]), np.array(Y[c], float))
            out.append(dict(sigma=sig, scen=scen, design=d, conflict=CONF[c], e_sig=np.log(s / true_sig),
                            e_pse=500 * (np.exp(m) - np.exp(true_mu[c])), n=len(X[c])))
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    js = [(s, w, sc, int(rng.integers(2**31))) for s, w in ((0.3, 0.3), (0.8, 0.7)) for sc in SCEN for _ in range(100)]
    with mp.get_context("fork").Pool(max(1, mp.cpu_count() - 1)) as p:
        df = pd.DataFrame([x for c in p.imap_unordered(one, js, chunksize=4) for x in c])
    df.to_csv(f"{SP}/drift_sim.csv", index=False)
    print("trials per condition:", df.groupby("design").n.mean().round(0).to_dict())
    df["conf"] = np.where(df.conflict.abs() > 0.3, "450", "0-250")
    g = df.groupby(["sigma", "scen", "design"], sort=False)
    t = pd.DataFrame({"sig_rmse%": 100 * np.sqrt(g.e_sig.apply(lambda e: np.mean(e**2))), "sig_bias%": 100 * g.e_sig.mean(),
                      "pse_rmse_ms": np.sqrt(g.e_pse.apply(lambda e: np.mean(e**2)))}).round(1)
    g2 = df[df.conf == "450"].groupby(["sigma", "scen", "design"], sort=False)
    t["pse_rmse_450"] = np.sqrt(g2.e_pse.apply(lambda e: np.mean(e**2))).round(1)
    t["sig_rmse_450%"] = (100 * np.sqrt(g2.e_sig.apply(lambda e: np.mean(e**2)))).round(1)
    print(t.to_string())
