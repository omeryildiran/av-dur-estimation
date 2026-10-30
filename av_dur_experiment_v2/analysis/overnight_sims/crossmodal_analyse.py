"""Cross-modal baseline: tables and figures for report/crossmodal_report.tex.

    python analysis/overnight_sims/crossmodal_analyse.py

Reads results/crossmodal_baseline.json (crossmodal_baseline.py) and the v1 real-data check
(results/v1_bias_fits.json and v1_baselines.json, from v1_bias_fits.py). Writes report/xm_*.pdf and
report/xm_numbers.json, and prints every table. Differences between fit variants are paired (the same
simulated participants and data), with 95% CIs from the per-participant differences.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RES, REP = HERE / "results", HERE / "report"
GENS = ["ci", "fusion", "switch"]
GEN_LABEL = {"ci": "causal inference", "fusion": "fusion", "switch": "switching"}
VARIANTS = ["ignore", "dfree", "xm500|same", "xm3|same", "xm3p|same", "xm500|none", "xm3|none", "xm3p|none", "oracle"]
LABEL = {"ignore": "ignore (c = d = 0)", "dfree": "d free, c = 0", "xm500": "500 ms block: c fixed",
         "xm3": "3 standards: c, d fixed", "xm3p": "3 standards: c, d priors", "oracle": "true c, d"}


def label(v):
    kind, _, tr = v.partition("|")
    return LABEL[kind] + ("" if not tr else f" ({'transfers' if tr == 'same' else 'no transfer'})")


SHORT = {"ignore": "ignore", "dfree": "d free", "xm500": "500 ms: c", "xm3": "3 std: c, d", "xm3p": "3 std: priors",
         "oracle": "true c, d"}


def short(v, sep=", "):
    """Figure label: 'ignore', '3 std: c, d', '3 std: c, d, no transfer', ..."""
    kind, _, tr = v.partition("|")
    return SHORT[kind] + (f"{sep}no transfer" if tr == "none" else "")


TITLE = {"ignore": "ignore (c = d = 0)", "xm3|same": "3 standards, c and d fixed\n(bias transfers)",
         "xm3|none": "3 standards, c and d fixed\n(no transfer)", "oracle": "true c and d"}


def load(path=RES / "crossmodal_baseline.json"):
    res = json.loads(path.read_text())
    fits, people = [], []
    for r in res:
        c1, c2, d = r["bias"]
        t = r["truth"]
        people.append(dict(gen=r["gen"], it=r["iter"], c1=c1, c2=c2, d=d, lam=t[0],
                           pc=t[3] if r["gen"] == "ci" else np.nan,
                           ps1=t[3] if r["gen"] == "switch" else np.nan, ps2=t[5] if r["gen"] == "switch" else np.nan,
                           **{f"x{n}_{tr}_{p}": r["xm"][tr][n]["est"][k] for tr in ("same", "none") for n in ("xm500", "xm3")
                              for k, p in enumerate(("c1", "c2", "d"))},
                           **{f"x{n}_{tr}_se_{p}": r["xm"][tr][n]["se"][k] for tr in ("same", "none") for n in ("xm500", "xm3")
                              for k, p in enumerate(("c1", "c2", "d"))},
                           **{f"xbias_none_{p}": r["xm"]["none"]["bias"][k] for k, p in enumerate(("c1", "c2", "d"))}))
        for variant, fv in r["fits"].items():
            for model, f in fv.items():
                p = f["params"]
                fits.append(dict(gen=r["gen"], it=r["iter"], variant=variant, model=model, aic=f["aic"], nll=f["nll"],
                                 f_pc=p.get("pc"), f_ps1=p.get("ps1"), f_ps2=p.get("ps2"), f_lam=p.get("lam"),
                                 f_c1=p.get("c1"), f_c2=p.get("c2"), f_d=p.get("d")))
    return pd.DataFrame(people), pd.DataFrame(fits)


def best_models(fits):
    b = fits.loc[fits.groupby(["gen", "it", "variant"]).aic.idxmin()].copy()
    b["hit"] = (b.model == b.gen).astype(float)
    return b


def paired(a, b):
    """Mean and 95% CI of a - b (per-participant differences)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    return m, m - 1.96 * se, m + 1.96 * se


def recovery_table(best):
    rows = []
    base = best[best.variant == "ignore"].set_index(["gen", "it"]).hit
    for v in VARIANTS:
        s = best[best.variant == v].set_index(["gen", "it"]).hit
        row = dict(variant=v, all=100 * s.mean(), **{g: 100 * s.xs(g).mean() for g in GENS})
        if v != "ignore":
            m, lo, hi = paired(s.loc[base.index], base)
            row.update(diff=100 * m, lo=100 * lo, hi=100 * hi)
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")


def rec(est, true):
    e, t = np.asarray(est, float), np.asarray(true, float)
    ok = np.isfinite(e) & np.isfinite(t)                  # a fit whose every start failed has NaN parameters
    e, t = e[ok], t[ok]
    return dict(r=np.corrcoef(e, t)[0, 1], rmse=np.sqrt(np.mean((e - t) ** 2)), bias=np.mean(e - t), sd_true=t.std(), n=len(e))


def param_tables(fits, people):
    m = fits.merge(people, on=["gen", "it"])
    out = {}
    ci = m[(m.gen == "ci") & (m.model == "ci")]
    out["pc"] = pd.DataFrame({v: rec(g.f_pc, g.pc) for v, g in ci.groupby("variant")}).T.loc[VARIANTS]
    base = ci[ci.variant == "ignore"].set_index("it")
    out["pc_paired"] = {}
    for v in VARIANTS[1:]:
        g = ci[ci.variant == v].set_index("it").loc[base.index]
        out["pc_paired"][v] = paired(np.abs(g.f_pc - g.pc), np.abs(base.f_pc - base.pc))
    sw = m[(m.gen == "switch") & (m.model == "switch")]
    out["ps"] = pd.DataFrame({v: rec(np.r_[g.f_ps1, g.f_ps2], np.r_[g.ps1, g.ps2]) for v, g in sw.groupby("variant")}).T.loc[VARIANTS]
    return out


def crossmodal_precision(people):
    rows = {}
    for n, ps in (("xm500", ("c1", "c2")), ("xm3", ("c1", "c2", "d"))):
        for p in ps:
            est, se = people[f"x{n}_same_{p}"], people[f"x{n}_same_se_{p}"]
            r = rec(est, people[p])
            r["coverage"] = np.mean(np.abs(est - people[p]) < 1.96 * 1.2 * se)       # with the priors' 1.2 x SE
            rows[f"{n} {p}"] = r
    return pd.DataFrame(rows).T


def by_bias_size(best, fits, people):
    m = best.merge(people, on=["gen", "it"])
    m["c_mean"] = (m.c1 + m.c2) / 2
    m["d_bin"] = pd.cut(m.d, [-0.31, 0.1, 0.3, 0.5, 0.71], labels=["< 0.1", "0.1-0.3", "0.3-0.5", "> 0.5"])
    m["c_bin"] = pd.cut(m.c_mean, [-0.51, 0.1, 0.3, 0.5, 1.01], labels=["< 0.1", "0.1-0.3", "0.3-0.5", "> 0.5"])
    sel = ["ignore", "xm3|same", "oracle"]
    rec_d = 100 * m[m.variant.isin(sel)].pivot_table(index="d_bin", columns="variant", values="hit", aggfunc="mean")[sel]
    rec_c = 100 * m[m.variant.isin(sel)].pivot_table(index="c_bin", columns="variant", values="hit", aggfunc="mean")[sel]
    n_d = m[m.variant == "ignore"].groupby("d_bin").size()
    n_c = m[m.variant == "ignore"].groupby("c_bin").size()
    ci = fits[(fits.gen == "ci") & (fits.model == "ci")].merge(people, on=["gen", "it"])
    ci["err"] = ci.f_pc - ci.pc
    ci["d_bin"] = pd.cut(ci.d, [-0.31, 0.1, 0.3, 0.5, 0.71], labels=["< 0.1", "0.1-0.3", "0.3-0.5", "> 0.5"])
    pc_d = ci[ci.variant.isin(sel)].pivot_table(index="d_bin", columns="variant", values="err", aggfunc="mean")[sel]
    return dict(rec_d=rec_d.assign(n=n_d), rec_c=rec_c.assign(n=n_c), pc_err_d=pc_d)


def evidence(fits):
    """How much the data say about the bias: AIC of the generating model with the true bias minus ignoring it."""
    own = fits[fits.model == fits.gen].pivot_table(index=["gen", "it"], columns="variant", values="nll")
    gain = 2 * (own["ignore"] - own["oracle"])          # same parameter count, so this is the AIC difference
    return gain


def v1_check():
    path = RES / "v1_bias_fits.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    out = {}
    for scn in sorted({r["scenario"] for r in res}):
        rows = [r for r in res if r["scenario"] == scn]
        best = {v: [min(r["fits"], key=lambda m: r["fits"][m][v]["aic"]) for r in rows] for v in ("nobias", "xc", "xc_d")}
        aic = {v: np.array([min(f[v]["aic"] for f in r["fits"].values()) for r in rows]) for v in ("nobias", "xc", "xc_d")}
        pc = {v: np.array([r["fits"]["ci"][v]["params"]["pc"] for r in rows]) for v in ("nobias", "xc", "xc_d")}
        out[scn] = dict(n=len(rows), pids=[r["pid"] for r in rows], best=best,
                        same_xc=int(sum(a == b for a, b in zip(best["nobias"], best["xc"]))),
                        same_xc_d=int(sum(a == b for a, b in zip(best["nobias"], best["xc_d"]))),
                        daic_xc=float((aic["xc"] - aic["nobias"]).sum()), daic_xc_d=float((aic["xc_d"] - aic["nobias"]).sum()),
                        max_dpc_xc=float(np.max(np.abs(pc["xc"] - pc["nobias"]))),
                        max_dpc_xc_d=float(np.max(np.abs(pc["xc_d"] - pc["nobias"]))))
    return out


def figures(best, fits, people, tables, sizes, gain):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    REP.mkdir(exist_ok=True)
    kinds = ["ignore", "dfree", "xm500", "xm3", "xm3p", "oracle"]
    col = {"ignore": "0.55", "dfree": "tab:purple", "xm500": "tab:orange", "xm3": "tab:blue", "xm3p": "tab:green", "oracle": "k"}

    # 1. model recovery by variant and generating model
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.3), sharey=True)
    for k, gsel in enumerate([["ci"], ["fusion"], ["switch"], GENS]):
        ax = axes[k]
        s = best[best.gen.isin(gsel)]
        x = 0
        ticks = []
        for kind in kinds:
            for tr in (["same", "none"] if kind.startswith("xm") else [None]):
                v = kind if tr is None else f"{kind}|{tr}"
                h = s[s.variant == v].hit
                m, se = 100 * h.mean(), 100 * h.std(ddof=1) / np.sqrt(len(h))
                ax.bar(x, m, 0.8, color=col[kind], alpha=1.0 if tr != "none" else 0.45, hatch="//" if tr == "none" else None,
                       edgecolor="white", lw=0)
                ax.errorbar(x, m, 1.96 * se, color="0.2", lw=0.8, capsize=1.5)
                ticks.append((x, short(v)))
                x += 1
            x += 0.4
        ax.set_xticks([t for t, _ in ticks])
        ax.set_xticklabels([lab for _, lab in ticks], rotation=60, ha="right", fontsize=7)
        ax.set(title="all" if len(gsel) == 3 else f"generated by {GEN_LABEL[gsel[0]]}", ylim=(40, 100))
        ax.axhline(100 / 3, color="0.85", lw=0.8)
    axes[0].set_ylabel("right model (AIC), %")
    fig.tight_layout()
    fig.savefig(REP / "xm_modelrec.pdf")
    plt.close(fig)

    # 2. p_c: fitted vs true for CI participants
    m = fits[(fits.gen == "ci") & (fits.model == "ci")].merge(people, on=["gen", "it"])
    sel = ["ignore", "xm3|same", "xm3|none", "oracle"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharex=True, sharey=True)
    for ax, v in zip(axes, sel):
        g = m[m.variant == v]
        ax.plot((0, 1), (0, 1), color="0.7", lw=0.8)
        sc = ax.scatter(g.pc, g.f_pc, c=g.d, cmap="coolwarm", vmin=-0.3, vmax=0.7, s=9, lw=0)
        r = rec(g.f_pc, g.pc)
        ax.text(0.03, 0.97, f"RMSE {r['rmse']:.3f}\nr = {r['r']:.2f}", transform=ax.transAxes, va="top", fontsize=8)
        ax.set(xlabel="true $p_c$", xlim=(0, 1), ylim=(-0.02, 1.02))
        ax.set_title(TITLE[v], fontsize=8.5)
    axes[0].set_ylabel("fitted $p_c$")
    fig.colorbar(sc, ax=axes, shrink=0.8, label="true d")
    fig.savefig(REP / "xm_pc.pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. by bias size
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    for ax, key, xl in ((axes[0], "rec_d", "true d"), (axes[1], "rec_c", "true c (mean of the two noise levels)")):
        t = sizes[key]
        for v in ("ignore", "xm3|same", "oracle"):
            ax.plot(range(len(t)), t[v], "o-", color=col[v.split("|")[0]], label=label(v))
        ax.set_xticks(range(len(t)))
        ax.set_xticklabels([f"{i}\n(n = {n})" for i, n in zip(t.index, t.n)], fontsize=7)
        ax.set(xlabel=xl, ylabel="right model (AIC), %", ylim=(40, 100))
    t = sizes["pc_err_d"]
    for v in ("ignore", "xm3|same", "oracle"):
        axes[2].plot(range(len(t)), t[v], "o-", color=col[v.split("|")[0]])
    axes[2].axhline(0, color="0.8", lw=0.8)
    axes[2].set_xticks(range(len(t)))
    axes[2].set_xticklabels(t.index, fontsize=7)
    axes[2].set(xlabel="true d", ylabel="mean error of $p_c$ (CI participants)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(REP / "xm_bias_size.pdf")
    plt.close(fig)

    # 4. what the cross-modal block measures
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.7))
    for ax, (n, p, title) in zip(axes, (("xm500", "c1", "500 ms block: c, low noise"), ("xm500", "c2", "500 ms block: c, high noise"),
                                         ("xm3", "c1", "3 standards: c, low noise"), ("xm3", "c2", "3 standards: c, high noise"),
                                         ("xm3", "d", "3 standards: d"))):
        est = people[f"x{n}_same_{p}"]
        lim = (-0.4, 1.0) if p != "d" else (-0.5, 0.9)
        ax.plot(lim, lim, color="0.7", lw=0.8)
        ax.scatter(people[p], est, s=6, lw=0, alpha=0.6)
        r = rec(est, people[p])
        ax.text(0.03, 0.97, f"RMSE {r['rmse']:.3f}\nr = {r['r']:.2f}", transform=ax.transAxes, va="top", fontsize=8)
        ax.set(title=title, xlabel="true", xlim=lim, ylim=lim)
    axes[0].set_ylabel("cross-modal estimate")
    fig.tight_layout()
    fig.savefig(REP / "xm_crossmodal.pdf")
    plt.close(fig)

    # 5. confusion matrices
    sel = ["ignore", "xm3|same", "xm3|none", "oracle"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.7))
    for ax, v in zip(axes, sel):
        s = best[best.variant == v]
        cm = pd.crosstab(s.gen, s.model, normalize="index").reindex(index=GENS, columns=GENS).fillna(0) * 100
        ax.imshow(cm.values, cmap="Blues", vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm.values[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if cm.values[i, j] > 60 else "black")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(["CI", "fusion", "switch"]); ax.set_yticklabels(["CI", "fusion", "switch"] if v == sel[0] else [])
        ax.set(xlabel="selected (AIC)")
        ax.set_title(TITLE[v], fontsize=8.5)
    axes[0].set_ylabel("generated")
    fig.tight_layout()
    fig.savefig(REP / "xm_confusion.pdf", bbox_inches="tight")
    plt.close(fig)

    # 6. evidence for the bias in the bimodal data
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    p = people.set_index(["gen", "it"]).loc[gain.index]
    for g in GENS:
        sel_g = gain.index.get_level_values(0) == g
        ax.scatter(p.d[sel_g], gain[sel_g], s=7, lw=0, alpha=0.7, label=GEN_LABEL[g])
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set(xlabel="true d", ylabel="AIC gain from the true bias")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(REP / "xm_evidence.pdf")
    plt.close(fig)


def main():
    people, fits = load()
    n = people.groupby("gen").size().to_dict()
    best = best_models(fits)
    rt = recovery_table(best)
    pt = param_tables(fits, people)
    xp = crossmodal_precision(people)
    sizes = by_bias_size(best, fits, people)
    gain = evidence(fits)
    v1 = v1_check()
    figures(best, fits, people, pt, sizes, gain)
    pd.set_option("display.width", 200)
    print(f"{len(people)} simulated participants: {n}\n")
    print("== bias drawn ==")
    print(people[["c1", "c2", "d"]].describe().round(2).to_string())
    print("\n== cross-modal estimates vs truth (transfer 'same'); coverage of +-1.96 x 1.2 SE ==")
    print(xp.round(3).to_string())
    print("\n== model recovery (AIC), % right; diff = paired difference from 'ignore' (95% CI) ==")
    print(rt.round(1).to_string())
    print("\n== p_c, CI participants (CI fit) ==")
    print(pt["pc"].astype(float).round(3).to_string())
    print("   paired change in |error| vs ignore (mean, 95% CI): " + "; ".join(f"{v} {m:+.3f} ({lo:+.3f}, {hi:+.3f})"
                                                                         for v, (m, lo, hi) in pt["pc_paired"].items()))
    print("\n== p_switch, switching participants (switching fit, both noise levels) ==")
    print(pt["ps"].astype(float).round(3).to_string())
    print("\n== model recovery by true d and by true c ==")
    print(sizes["rec_d"].round(0).to_string()); print(sizes["rec_c"].round(0).to_string())
    print("\n== mean p_c error by true d (CI participants) ==")
    print(sizes["pc_err_d"].round(3).to_string())
    print("\n== AIC gain of the generating model with the true bias over ignoring it ==")
    print(gain.groupby(level=0).describe(percentiles=[0.5, 0.9]).round(1).to_string())
    print(f"   share with a gain > 2: {(gain > 2).mean():.2f}; > 10: {(gain > 10).mean():.2f}")
    if v1:
        print("\n== v1 real data (11 participants): c fixed at the cross-modal value (xc), and with d free (xc_d) ==")
        for scn, o in v1.items():
            print(f"-- {scn}: same winning model as without bias: xc {o['same_xc']}/{o['n']}, xc_d {o['same_xc_d']}/{o['n']}; "
                  f"summed AIC change xc {o['daic_xc']:+.0f}, xc_d {o['daic_xc_d']:+.0f}; "
                  f"largest p_c change xc {o['max_dpc_xc']:.2f}, xc_d {o['max_dpc_xc_d']:.2f}")
    numbers = dict(n=n, n_total=len(people), recovery=rt.round(2).to_dict(orient="index"),
                   pc=pt["pc"].astype(float).round(4).to_dict(orient="index"),
                   pc_paired={v: list(map(float, x)) for v, x in pt["pc_paired"].items()},
                   ps=pt["ps"].astype(float).round(4).to_dict(orient="index"), crossmodal=xp.astype(float).round(4).to_dict(orient="index"),
                   rec_d=sizes["rec_d"].round(2).to_dict(orient="index"), rec_c=sizes["rec_c"].round(2).to_dict(orient="index"),
                   pc_err_d=sizes["pc_err_d"].round(4).to_dict(orient="index"),
                   gain=dict(median=float(gain.median()), share_gt2=float((gain > 2).mean()), share_gt10=float((gain > 10).mean())),
                   v1=v1)
    (REP / "xm_numbers.json").write_text(json.dumps(numbers, indent=1, default=str))


if __name__ == "__main__":
    main()
