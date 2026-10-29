"""Figures and numbers for the overnight simulation report (results/ -> report/)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, REP = HERE / "results", HERE / "report"
REP.mkdir(exist_ok=True)
S_PER_TRIAL = 3.1          # measured: P101-P104 sessions incl. cues and breaks (P102 interleaved cross-modal 3.09 s)
INK, BLUE, ORANGE, GREEN, PURPLE, GREY = "#1F2430", "#2F6DB5", "#D9822B", "#5B8C5A", "#8E5BA8", "#888888"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "legend.frameon": False})
NUM = {}


def save(fig, name):
    fig.savefig(REP / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


# ── 1. unimodal and cross-modal: staircases vs QUEST+ ────────────────────────────
def adaptive():
    d = pd.read_csv(RES / "adaptive_raw.csv")
    d["e_ms"] = 500 * (np.exp(d.mu_hat) - np.exp(d.true_mu))
    d["e_mu"] = d.mu_hat - d.true_mu
    d["e_ls"] = np.log(d.sigma_hat / d.true_sigma)
    g = d.groupby(["task", "variant", "method", "budget"])
    s = pd.DataFrame(dict(pse_rmse_ms=g.e_ms.apply(lambda e: np.sqrt(np.mean(e ** 2))),
                          pse_sd_log=g.e_mu.std(), pse_bias_ms=g.e_ms.mean(),
                          pse_med_ms=g.e_ms.apply(lambda e: np.median(np.abs(e))),
                          pse_p90_ms=g.e_ms.apply(lambda e: np.percentile(np.abs(e), 90)),
                          pse_fail=g.e_ms.apply(lambda e: np.mean(np.abs(e) > 150)),
                          pse_sd_log_robust=g.e_mu.apply(lambda e: 1.4826 * np.median(np.abs(e - np.median(e)))),
                          sig_rmse=g.e_ls.apply(lambda e: np.sqrt(np.mean(e ** 2))),
                          sig_bias=g.e_ls.mean(),
                          sig_fail=g.sigma_hat.apply(lambda v: np.mean(v > 2.9)), n=g.size())).reset_index()
    s.to_csv(REP / "adaptive_summary.csv", index=False)
    col = {"st4": GREY, "stxm": GREY, "st11": GREEN, "q_ms": BLUE, "q_mu": ORANGE}
    lab = {"st4": "4 staircases (old unimodal)", "stxm": "4 staircases (current cross-modal)", "st11": "two 1D1U (300/800 design)",
           "q_ms": "QUEST+ PSE & slope", "q_mu": "QUEST+ PSE only"}
    titles = {"aud_low": "Auditory, low noise", "aud_high": "Auditory, high noise", "visual": "Visual",
              "xm_low": "Cross-modal, low noise", "xm_high": "Cross-modal, high noise"}
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.4))
    for j, task in enumerate(titles):
        h = s[(s.task == task) & (s.variant == "base")]
        for m, hm in h.groupby("method"):
            axes[0, j].plot(hm.budget, hm.sig_rmse, "o-", color=col[m], ms=3.5, label=lab[m])
            axes[1, j].plot(hm.budget, hm.pse_med_ms, "o-", color=col[m], ms=3.5, label=lab[m])
            axes[1, j].plot(hm.budget, hm.pse_p90_ms, ":", color=col[m], lw=1)
        axes[0, j].set_title(titles[task]); axes[1, j].set_xlabel("Trials in the condition (incl. 7% easy)")
        axes[0, j].set_ylim(0, None); axes[1, j].set_ylim(0, None)
        axes[0, j].legend(fontsize=7.5, loc="upper right")
    axes[0, 0].set_ylabel("σ error, RMSE of ln(σ̂/σ)"); axes[1, 0].set_ylabel("PSE error (ms at 500 ms)\nmedian (solid), 90th percentile (dotted)")
    fig.tight_layout(); save(fig, "fig_adaptive")

    # robustness table at the budget each block uses now
    rob = s[s.task.isin(["aud_high", "xm_low"]) & s.budget.isin([100, 200])]
    rob.to_csv(REP / "adaptive_robustness.csv", index=False)

    def at(task, method, budget, col_, variant="base"):
        r = s[(s.task == task) & (s.method == method) & (s.budget == budget) & (s.variant == variant)]
        return float(r[col_].iloc[0]) if len(r) else float("nan")

    def budget_for(task, method, target, col_):
        h = s[(s.task == task) & (s.method == method) & (s.variant == "base")].sort_values("budget")
        y = h[col_].to_numpy(); x = h.budget.to_numpy()
        if y[-1] > target:
            return float("nan")
        i = np.argmax(y <= target)
        if i == 0:
            return float(x[0])
        return float(np.interp(target, [y[i], y[i - 1]], [x[i], x[i - 1]]))

    out = {}
    for task in ("aud_low", "aud_high", "visual"):
        ref = at(task, "st4", 150, "sig_rmse")
        out[task] = dict(q_ms_budget_same_sig=budget_for(task, "q_ms", ref, "sig_rmse"),
                         **{f"{m}_{b}_pse_med": at(task, m, b, "pse_med_ms") for m in ("st4", "q_ms", "q_mu") for b in (60, 100, 150)},
                         **{f"{m}_{b}_pse_fail": at(task, m, b, "pse_fail") for m in ("st4", "q_ms", "q_mu") for b in (100, 150)},
                         **{f"{m}_{b}_sig": at(task, m, b, "sig_rmse") for m in ("st4", "q_ms", "q_mu") for b in (60, 80, 100, 120, 150)},
)
    for task in ("xm_low", "xm_high"):
        ref = at(task, "stxm", 200, "pse_med_ms")
        out[task] = dict(stxm_200_pse=ref, **{f"{m}_{b}_pse_med": at(task, m, b, "pse_med_ms") for m in ("stxm", "st11", "q_ms", "q_mu") for b in (60, 100, 150, 200)},
                         **{f"{m}_{b}_pse_fail": at(task, m, b, "pse_fail") for m in ("stxm", "st11", "q_ms", "q_mu") for b in (100, 200)},
                         budget_same_pse={m: budget_for(task, m, ref, "pse_med_ms") for m in ("stxm", "st11", "q_ms", "q_mu")},
                         sig_rmse_200={m: at(task, m, 200, "sig_rmse") for m in ("stxm", "st11", "q_ms", "q_mu")})
    NUM["adaptive"] = out
    return s


# ── 2. cross-modal standards: precision of the growth term d ──────────────────────
def d_precision(s):
    """SE of d (slope of the log bias against ln(standard/500)) for one participant and noise level,
    from the simulated SD of the fitted ln-PSE at each standard's trial count."""
    x = np.log(np.array([0.3, 0.5, 0.8]) / 0.5)
    rows = []
    for task in ("xm_low", "xm_high"):
        for method in ("stxm", "st11", "q_mu"):
            h = s[(s.task == task) & (s.method == method) & (s.variant == "base")].sort_values("budget")
            sd = lambda n: float(np.interp(n, h.budget, h.pse_sd_log_robust))  # noqa: E731
            for name, alloc in (("current 86/194/86", (86, 194, 86)), ("equal 122/122/122", (122, 122, 122)),
                                ("short 60/120/60", (60, 120, 60)), ("long 120/200/120", (120, 200, 120))):
                w = 1 / np.array([sd(n) ** 2 for n in alloc])
                xb = np.sum(w * x) / w.sum()
                se_d = 1 / np.sqrt(np.sum(w * (x - xb) ** 2))
                se_c = 1 / np.sqrt(w.sum()) if False else sd(alloc[1])
                rows.append(dict(task=task, method=method, allocation=name, trials=sum(alloc), se_d=se_d, se_c500=se_c,
                                 minutes=sum(alloc) * 2 * S_PER_TRIAL / 60))
    t = pd.DataFrame(rows)
    t.to_csv(REP / "d_precision.csv", index=False)
    NUM["d"] = t.round(3).to_dict("records")
    return t


# ── 3. main experiment: model recovery ─────────────────────────────────────────────
NAMES = {"ci": "CI", "fusion": "Fusion", "switch": "Switch"}
ORDER = ("ci", "fusion", "switch")


def bimodal():
    path = RES / "bimodal_recovery.json"
    R = json.loads(path.read_text())
    designs = list(R[0]["designs"])
    rows, prow, pse_rows = [], [], []
    for r in R:
        for dn, dv in r["designs"].items():
            f = dv["fits"]
            for crit in ("aic", "bic"):
                win = min(ORDER, key=lambda m: f[m][crit])
                rows.append(dict(gen=r["gen"], iter=r["iter"], design=dn, crit=crit, win=win, n_trials=dv["n_trials"],
                                 ci_worse=f["ci"]["nll"] > f["fusion"]["nll"] + 0.05,
                                 d_aic_ci_fus=f["fusion"]["aic"] - f["ci"]["aic"]))
            if r["gen"] == "ci":
                t = r["truth"]; p = f["ci"]["params"]
                prow.append(dict(design=dn, pc_true=t[3], pc_hat=p[3], sa1=t[1], sa1_hat=p[1], sv=t[2], sv_hat=p[2],
                                 sa2=t[4], sa2_hat=p[4], lam=t[0], lam_hat=p[0]))
            for q in dv["pse"]:
                pse_rows.append(dict(design=dn, gen=r["gen"], **q))
    W = pd.DataFrame(rows); P = pd.DataFrame(prow); Q = pd.DataFrame(pse_rows)
    NUM["bimodal_n"] = {g: int((W[(W.design == designs[0]) & (W.crit == "aic")].gen == g).sum()) for g in ORDER}
    NUM["ci_worse_than_fusion"] = int(W[W.crit == "aic"].ci_worse.sum())
    summ = []
    for (dn, crit), h in W.groupby(["design", "crit"]):
        acc = {g: float((h[h.gen == g].win == g).mean()) for g in ORDER}
        summ.append(dict(design=dn, crit=crit, n_trials=int(h.n_trials.iloc[0]), minutes=h.n_trials.iloc[0] * S_PER_TRIAL / 60,
                         mean_acc=float(np.mean(list(acc.values()))), **{f"acc_{g}": v for g, v in acc.items()}))
    summ = pd.DataFrame(summ).sort_values(["crit", "n_trials"])
    summ.to_csv(REP / "bimodal_summary.csv", index=False)
    NUM["bimodal"] = summ.round(3).to_dict("records")

    # confusion matrices (AIC), one per design
    order_d = sorted(designs, key=lambda n: (n.split("_")[0], "_w" in n, int(n.split("_")[-1])))
    fig, axes = plt.subplots(1, len(order_d), figsize=(2.55 * len(order_d), 2.9))
    for ax, dn in zip(axes, order_d):
        h = W[(W.design == dn) & (W.crit == "aic")]
        M = np.array([[np.mean(h[h.gen == g].win == w) for w in ORDER] for g in ORDER])
        ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", color="white" if M[i, j] > 0.55 else INK, fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels([NAMES[m] for m in ORDER], fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels([NAMES[m] for m in ORDER] if ax is axes[0] else [], fontsize=8)
        n = int(h.n_trials.iloc[0])
        ax.set_title(f"{dn.replace('_', ' ')}\n{n} trials, {n * S_PER_TRIAL / 60:.0f} min", fontsize=8.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("Generating model")
    fig.text(0.5, -0.04, "Selected model (AIC); % of simulated participants", ha="center", fontsize=9)
    fig.tight_layout(); save(fig, "fig_confusion")

    # accuracy vs session length
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, crit in enumerate(("aic", "bic")):
        h = summ[summ.crit == crit]
        for base, c, mk in (("9c_", BLUE, "o"), ("7c_", GREY, "s"), ("9c_w", ORANGE, "D")):
            hh = h[h.design.str.startswith(base) & (~h.design.str.contains("_w") if base != "9c_w" else True)]
            for g, ls in (("mean_acc", "-"),):
                ax[k].plot(hh.minutes, 100 * hh[g], mk + ls, color=c, label={"9c_": "9 conflicts ±450", "7c_": "7 conflicts ±250", "9c_w": "9 conflicts, more trials at large conflicts"}[base])
        ax[k].set(xlabel="Main experiment length (min, 3.1 s per trial)", ylabel="Correct model (%, mean of 3)",
                  title=f"Model recovery ({crit.upper()})", ylim=(30, 100))
        ax[k].axhline(100 / 3, color="0.8", lw=1)
        ax[k].legend(fontsize=8)
    fig.tight_layout(); save(fig, "fig_recovery_vs_time")

    # CI parameter recovery (p_c, sigmas) per design
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
    for ax, (tcol, hcol, title) in zip(axes, (("pc_true", "pc_hat", "p_c"), ("sa1", "sa1_hat", "σa low"),
                                               ("sa2", "sa2_hat", "σa high"), ("sv", "sv_hat", "σv"))):
        for dn, c in (("9c_60", ORANGE), ("9c_154", BLUE), ("7c_154", GREY)):
            h = P[P.design == dn]
            if len(h):
                ax.scatter(h[tcol], h[hcol], s=12, color=c, alpha=0.7, label=dn.replace("_", " "))
        lim = (0, 1) if tcol == "pc_true" else (0, 1.3)
        ax.plot(lim, lim, color="0.7", lw=1); ax.set(xlim=lim, ylim=lim, xlabel="true", ylabel="fitted (CI)", title=title)
    axes[0].legend(fontsize=7.5)
    fig.tight_layout(); save(fig, "fig_ci_params")
    prec = []
    for dn, h in P.groupby("design"):
        prec.append(dict(design=dn, pc_rmse=float(np.sqrt(np.mean((h.pc_hat - h.pc_true) ** 2))),
                         pc_r=float(np.corrcoef(h.pc_true, h.pc_hat)[0, 1]),
                         sa1_rmse_ln=float(np.sqrt(np.mean(np.log(h.sa1_hat / h.sa1) ** 2))),
                         sa2_rmse_ln=float(np.sqrt(np.mean(np.log(h.sa2_hat / h.sa2) ** 2))),
                         sv_rmse_ln=float(np.sqrt(np.mean(np.log(h.sv_hat / h.sv) ** 2)))))
    NUM["ci_params"] = pd.DataFrame(prec).round(3).to_dict("records")

    # recovery of CI by true p_c
    by_pc = []
    for r in R:
        if r["gen"] != "ci":
            continue
        for dn, dv in r["designs"].items():
            f = dv["fits"]
            by_pc.append(dict(design=dn, pc=r["truth"][3], ok=min(ORDER, key=lambda m: f[m]["aic"]) == "ci"))
    B = pd.DataFrame(by_pc)
    B["bin"] = pd.cut(B.pc, [0, 0.35, 0.65, 0.9])
    NUM["ci_by_pc"] = {dn: {str(k): float(v) for k, v in h.groupby("bin", observed=True).ok.mean().items()} for dn, h in B.groupby("design")}

    # PSE precision per condition
    Q["err_ms"] = 1000 * (Q.pse_hat - Q.pse_true)
    Q = Q[np.isfinite(Q.err_ms)]
    Q["absc"] = np.round(np.abs(Q.conflict) * 1000).astype(int)
    pq = Q.groupby(["design", "noise"]).err_ms.apply(lambda e: float(np.sqrt(np.mean(np.clip(e, -400, 400) ** 2)))).unstack()
    NUM["pse_rmse_ms"] = {dn: {int(k): float(v) for k, v in row.items()} for dn, row in pq.iterrows()}
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for k, ni in enumerate((0, 1)):
        for dn, c in (("9c_60", ORANGE), ("9c_90", GREEN), ("9c_120", PURPLE), ("9c_154", BLUE)):
            h = Q[(Q.design == dn) & (Q.noise == ni)].groupby("absc").err_ms.apply(lambda e: np.sqrt(np.mean(np.clip(e, -400, 400) ** 2)))
            ax[k].plot(h.index, h.values, "o-", color=c, label=dn.replace("9c_", "") + " trials")
        ax[k].set(title=("Low" if ni == 0 else "High") + " noise", xlabel="|conflict| (ms)")
    ax[0].set_ylabel("PSE error per condition, RMSE (ms)"); ax[0].legend(fontsize=8)
    fig.tight_layout(); save(fig, "fig_pse_precision")
    return summ


def fixed_sv():
    """Same participants and answers, sigma_V fitted (bimodal_recovery) vs fixed from the unimodal block (fixed_sv)."""
    F = json.loads((RES / "fixed_sv.json").read_text())
    R = {(r["gen"], r["iter"]): r for r in json.loads((RES / "bimodal_recovery.json").read_text())}
    rows = []
    for r in F:
        base = R.get((r["gen"], r["iter"]))
        if base is None:
            continue
        assert np.allclose(base["truth"], r["truth"])
        for dn, dv in r["designs"].items():
            for how, f in (("fitted", base["designs"][dn]["fits"]), ("fixed", dv["fits"])):
                for crit in ("aic", "bic"):
                    win = min(ORDER, key=lambda m: f[m][crit])
                    row = dict(gen=r["gen"], iter=r["iter"], design=dn, how=how, crit=crit, ok=win == r["gen"])
                    if r["gen"] == "ci":
                        row.update(pc_true=r["truth"][3], pc_hat=f["ci"]["params"][3],
                                   sa1_err=np.log(f["ci"]["params"][1] / r["truth"][1]), sa2_err=np.log(f["ci"]["params"][4] / r["truth"][4]))
                    rows.append(row)
    D = pd.DataFrame(rows)
    out = []
    for (dn, how, crit), h in D.groupby(["design", "how", "crit"]):
        acc = {g: float(h[h.gen == g].ok.mean()) for g in ORDER}
        c = h[h.gen == "ci"]
        out.append(dict(design=dn, sigma_v=how, crit=crit, n_per_model=int(len(h) / 3), mean_acc=float(np.mean(list(acc.values()))),
                        **{f"acc_{g}": v for g, v in acc.items()},
                        pc_rmse=float(np.sqrt(np.mean((c.pc_hat - c.pc_true) ** 2))), pc_r=float(np.corrcoef(c.pc_true, c.pc_hat)[0, 1]),
                        sa1_rmse_ln=float(np.sqrt(np.mean(c.sa1_err ** 2))), sa2_rmse_ln=float(np.sqrt(np.mean(c.sa2_err ** 2)))))
    out = pd.DataFrame(out)
    out.to_csv(REP / "fixed_sv_summary.csv", index=False)
    NUM["fixed_sv"] = out.round(3).to_dict("records")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    c = D[(D.gen == "ci") & (D.crit == "aic")]
    for ax, dn in zip(axes, ("7c_90", "9c_60", "9c_90")):
        for how, col in (("fitted", GREY), ("fixed", BLUE)):
            h = c[(c.design == dn) & (c.how == how)]
            ax.scatter(h.pc_true, h.pc_hat, s=12, color=col, alpha=0.75,
                       label=f"σV {'fitted' if how == 'fitted' else 'from unimodal'}: RMSE {np.sqrt(np.mean((h.pc_hat - h.pc_true) ** 2)):.2f}")
        ax.plot((0, 1), (0, 1), color="0.7", lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="true p_c", title=dn.replace("_", " ") + " trials per condition")
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("fitted p_c (CI)")
    fig.tight_layout(); save(fig, "fig_fixed_sv")
    return out


QNAMES = {"stair": "Staircases (2D1U + 1D2U)", "q_ms": "QUEST+ PSE & slope", "q_mu": "QUEST+ PSE only"}


def qplus():
    """Main task, 9 conflicts x 72 trials: staircases vs QUEST+ per condition (sigma_V as in a joint fit)."""
    R = json.loads((RES / "qplus_main.json").read_text())
    meths = list(QNAMES)
    rows, pse = [], []
    for r in R:
        for meth, mv in r["methods"].items():
            f = mv["fits"]
            row = dict(gen=r["gen"], iter=r["iter"], method=meth,
                       ok_aic=min(ORDER, key=lambda m: f[m]["aic"]) == r["gen"], ok_bic=min(ORDER, key=lambda m: f[m]["bic"]) == r["gen"],
                       win_aic=min(ORDER, key=lambda m: f[m]["aic"]))
            g = f[r["gen"]]["params"]; t = r["truth"]
            row.update(sa1_err=np.log(g[1] / t[1]), sa2_err=np.log(g[4 if r["gen"] != "fusion" else 3] / t[4 if r["gen"] != "fusion" else 3]),
                       lam_err=g[0] - t[0])
            if r["gen"] == "ci":
                row.update(pc_true=t[3], pc_hat=g[3])
            if r["gen"] == "switch":
                row.update(ps1_err=g[3] - t[3], ps2_err=g[5] - t[5])
            rows.append(row)
            for q in mv["pse"]:
                pse.append(dict(method=meth, gen=r["gen"], **q))
    D = pd.DataFrame(rows); Q = pd.DataFrame(pse)
    rng = np.random.default_rng(0)
    summ = []
    for meth in meths:
        h = D[D.method == meth]
        c = h[h.gen == "ci"]; sw = h[h.gen == "switch"]
        row = dict(method=meth, n_per_model=int(len(h) / 3))
        for crit in ("aic", "bic"):
            acc = {g: float(h[h.gen == g][f"ok_{crit}"].mean()) for g in ORDER}
            row.update({f"{crit}_{g}": v for g, v in acc.items()}, **{f"{crit}_mean": float(np.mean(list(acc.values())))})
        row.update(pc_rmse=float(np.sqrt(np.mean((c.pc_hat - c.pc_true) ** 2))), pc_r=float(np.corrcoef(c.pc_true, c.pc_hat)[0, 1]),
                   # sigma_A for CI and fusion participants (weakly identified under switching with mostly visual answers)
                   sa1_rmse_ln=float(np.sqrt(np.mean(h[h.gen != "switch"].sa1_err ** 2))),
                   sa2_rmse_ln=float(np.sqrt(np.mean(h[h.gen != "switch"].sa2_err ** 2))),
                   pswitch_rmse=float(np.sqrt(np.mean(np.r_[sw.ps1_err, sw.ps2_err] ** 2))))
        summ.append(row)
    summ = pd.DataFrame(summ)
    # paired differences vs the staircases (bootstrap over simulated participants)
    W = D.pivot_table(index=["gen", "iter"], columns="method", values="ok_aic").dropna()
    C = D[D.gen == "ci"].pivot_table(index="iter", columns="method", values="pc_hat").dropna()
    pct = D[D.gen == "ci"].groupby("iter").pc_true.first().loc[C.index]
    paired = {}
    for meth in meths[1:]:
        dacc, dpc = [], []
        for _ in range(2000):
            i = rng.integers(len(W), size=len(W)); j = rng.integers(len(C), size=len(C))
            dacc.append(W[meth].values[i].mean() - W["stair"].values[i].mean())
            e = lambda m: np.sqrt(np.mean((C[m].values[j] - pct.values[j]) ** 2))  # noqa: E731
            dpc.append(e(meth) - e("stair"))
        paired[meth] = dict(d_acc=float(W[meth].mean() - W["stair"].mean()), d_acc_ci=[float(v) for v in np.percentile(dacc, [2.5, 97.5])],
                            d_pc_rmse=float(np.mean(dpc)), d_pc_rmse_ci=[float(v) for v in np.percentile(dpc, [2.5, 97.5])])
    Q["err_ms"] = 1000 * (Q.pse_hat - Q.pse_true)
    Q = Q[np.isfinite(Q.err_ms)]
    Q["absc"] = np.round(np.abs(Q.conflict) * 1000).astype(int)
    rm = lambda e: float(np.sqrt(np.mean(np.clip(e, -400, 400) ** 2)))  # noqa: E731
    pq = Q.groupby(["method", "noise"]).err_ms.apply(rm).unstack()
    summ = summ.merge(pq.rename(columns={0: "pse_rmse_low", 1: "pse_rmse_high"}), left_on="method", right_index=True)
    summ.to_csv(REP / "qplus_main_summary.csv", index=False)
    NUM["qplus_main"] = dict(summary=summ.round(3).to_dict("records"), paired=paired)

    col = {"stair": GREY, "q_ms": BLUE, "q_mu": ORANGE}
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9))
    for ax, meth in zip(axes, meths):
        h = D[D.method == meth]
        Mx = np.array([[np.mean(h[h.gen == g].win_aic == w) for w in ORDER] for g in ORDER])
        ax.imshow(Mx, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{100 * Mx[i, j]:.0f}", ha="center", va="center", color="white" if Mx[i, j] > 0.55 else INK, fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels([NAMES[m] for m in ORDER], fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels([NAMES[m] for m in ORDER] if ax is axes[0] else [], fontsize=8)
        ax.set_title(QNAMES[meth], fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("Generating model")
    fig.text(0.5, -0.03, "Selected model (AIC); % of simulated participants", ha="center", fontsize=9)
    fig.tight_layout(); save(fig, "fig_qplus_confusion")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    for k, ni in enumerate((0, 1)):
        for meth in meths:
            h = Q[(Q.method == meth) & (Q.noise == ni)].groupby("absc").err_ms.apply(rm)
            ax[k].plot(h.index, h.values, "o-", color=col[meth], label=QNAMES[meth])
        ax[k].set(title=("Low" if ni == 0 else "High") + " noise: PSE per condition", xlabel="|conflict| (ms)", ylim=(0, None))
    ax[0].set_ylabel("PSE error, RMSE (ms)"); ax[0].legend(fontsize=8)
    for meth in meths:
        h = D[(D.method == meth) & (D.gen == "ci")]
        ax[2].scatter(h.pc_true, h.pc_hat, s=11, alpha=0.7, color=col[meth],
                      label=f"{QNAMES[meth]}: RMSE {np.sqrt(np.mean((h.pc_hat - h.pc_true) ** 2)):.2f}")
    ax[2].plot((0, 1), (0, 1), color="0.7", lw=1); ax[2].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="true p_c", ylabel="fitted p_c", title="CI participants")
    ax[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); save(fig, "fig_qplus_pse_pc")
    return summ


def main():
    s = adaptive() if (RES / "adaptive_raw.csv").exists() else None
    if s is not None:
        d_precision(s)
    if (RES / "bimodal_recovery.json").exists():
        bimodal()
    if (RES / "fixed_sv.json").exists():
        fixed_sv()
    if (RES / "qplus_main.json").exists():
        qplus()
    (REP / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
    print(json.dumps(NUM, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
