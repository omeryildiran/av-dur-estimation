"""Main experiment, one noise level: 9 conflicts x 102 staircase trials (918) + 96 easy (3 parts x 32).
Shared staircases (v1: 2 for all conditions) vs per-condition staircases; 2D1U+1D2U vs 2D1U+1D3U.
Observer: PSE(c) = ln(1 + w c / 0.5) (partial visual capture), sigma; per-condition fit (lapse <= 0.1)."""
import sys, multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/ofy204/Documents/my_repos/av-dur-estimation/av_dur_experiment_v2")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from two_stairs import qs
from avdur.staircase import Staircase
import experiment_settings as S
CONF = np.array(S.TASKS["main"]["blocks"]["part1"]); FLOOR = S.TASKS["main"]["floor"]; N_PER_COND = 102; N_EASY = 96
RULES = {"2D1U+1D2U": (("2D1U", "floor"), ("1D2U", "ceiling")), "2D1U+1D3U": (("2D1U", "floor"), ("1D3U", "ceiling"))}
DESIGNS = [(r, sh) for r in RULES for sh in ("shared", "per condition")]

def mk(rule, start):
    return Staircase(rule, FLOOR, S.STAIRCASE["ceiling"], init_step=S.STAIRCASE["init_step"], step_factor=S.STAIRCASE["step_factor"],
                     n_step_changes=S.STAIRCASE["n_step_changes"], start=start, start_top=S.STAIRCASE["start_top"])

def run(rules, shared, mus, sig, lapse, rng):
    nc = len(CONF)
    trials = [("s", c, k) for c in range(nc) for k in range(2) for _ in range(N_PER_COND // 2)]
    trials += [("e", int(rng.integers(nc)), i % 2) for i in range(N_EASY)]
    trials = [trials[i] for i in rng.permutation(len(trials))]
    stairs = {key: mk(*RULES[rules][key[1]]) for key in ([(0, k) for k in range(2)] if shared else [(c, k) for c in range(nc) for k in range(2)])}
    X = {c: [] for c in range(nc)}; Y = {c: [] for c in range(nc)}
    for kind, c, k in trials:
        if kind == "e":
            x = np.log(1 + (FLOOR if k == 0 else S.TEST_CEILING)); st = None
        else:
            st = stairs[(0, k) if shared else (c, k)]; x = np.log(1 + st.level)
        p = lapse / 2 + (1 - lapse) * qs.norm_cdf((x - mus[c]) / sig)
        y = int(rng.random() < p)
        if st is not None: st.update(y)
        X[c].append(x); Y[c].append(y)
    return X, Y

def one(a):
    sig, w, lapse, seed = a
    mus = np.log(1 + w * CONF / 0.5) + np.random.default_rng(seed).uniform(-0.1, 0.1)
    out = []
    for i, (r, sh) in enumerate(DESIGNS):
        rng = np.random.default_rng([seed, i])
        X, Y = run(r, sh == "shared", mus, sig, lapse, rng)
        for c in range(len(CONF)):
            m, s = qs.fit_ml(np.array(X[c]), np.array(Y[c], float))
            out.append(dict(sigma=sig, w=w, design=f"{sh}, {r}", conflict=CONF[c], e_sig=np.log(s / sig), e_pse=500 * (np.exp(m) - np.exp(mus[c]))))
    return out

if __name__ == "__main__":
    from scipy.stats import norm; qs.norm_cdf = norm.cdf
    rng = np.random.default_rng(5)
    js = [(s, w, l, int(rng.integers(2**31))) for s, w in ((0.3, 0.3), (0.8, 0.7)) for l in (0.0, 0.05) for _ in range(100)]
    with mp.get_context("fork").Pool(max(1, mp.cpu_count() - 1)) as p:
        df = pd.DataFrame([x for c in p.imap_unordered(one, js, chunksize=4) for x in c])
    df.to_csv(__import__("pathlib").Path(__file__).resolve().parent / "main_shared_sim.csv", index=False)
    df["extreme"] = np.where(df.conflict.abs() > 0.3, "+-450", "0..+-250")
    g = df.groupby(["sigma", "design", "extreme"])
    print(pd.DataFrame({"sigma_rmse_%": 100 * np.sqrt(g.e_sig.apply(lambda e: np.mean(e**2))),
                        "sigma_bias_%": 100 * g.e_sig.mean(),
                        "pse_rmse_ms": np.sqrt(g.e_pse.apply(lambda e: np.mean(e**2)))}).round(1).to_string())
