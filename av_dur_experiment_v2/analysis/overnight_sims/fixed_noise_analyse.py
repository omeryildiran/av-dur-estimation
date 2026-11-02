"""Tables and a figure for fixed_noise_sim.py: model recovery and p_c / p_switch error against the
high-noise psychometric sigma.

    python analysis/overnight_sims/fixed_noise_analyse.py     # results/fixed_noise_analyse.txt, report/fixed_noise.pdf/.png
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bimodal_recovery as B  # noqa: E402

LAB = {"ci": "causal inference", "fusion": "fusion", "switch": "switching"}


def main():
    res = [r for r in json.loads((B.OUT / "fixed_noise_sim.json").read_text()) if "fits" in r]
    psis = sorted({r["psi"] for r in res})
    lines, rec, pc_err, ps_err, aic_gap = [], {}, {}, {}, {}
    for psi in psis:
        rr = [r for r in res if r["psi"] == psi]
        for g in B.MODELS:
            g_rr = [r for r in rr if r["gen"] == g]
            win = [min(r["fits"], key=lambda m: r["fits"][m]["aic"]) for r in g_rr]
            rec[psi, g] = (np.mean([w == g for w in win]), len(g_rr))
            gaps = []
            for r in g_rr:
                a = sorted(f["aic"] for m, f in r["fits"].items() if m != g)
                gaps.append(a[0] - r["fits"][g]["aic"])
            aic_gap[psi, g] = np.median(gaps)
        ci = [r for r in rr if r["gen"] == "ci"]
        pc_err[psi] = np.sqrt(np.mean([(r["fits"]["ci"]["params"]["pc"] - r["truth"][3]) ** 2 for r in ci]))
        sw = [r for r in rr if r["gen"] == "switch"]
        ps_err[psi] = np.sqrt(np.mean([(r["fits"]["switch"]["params"]["ps2"] - r["truth"][5]) ** 2 for r in sw]))
        allrec = np.mean([rec[psi, g][0] for g in B.MODELS])
        lines.append(f"psi {psi:3.1f}: recovery {100 * allrec:5.1f}%  (" +
                     ", ".join(f"{LAB[g]} {100 * rec[psi, g][0]:3.0f}% n {rec[psi, g][1]}" for g in B.MODELS) +
                     f")  median AIC margin " + "/".join(f"{aic_gap[psi, g]:.0f}" for g in B.MODELS) +
                     f"  p_c RMSE {pc_err[psi]:.3f}  p_switch(high) RMSE {ps_err[psi]:.3f}")
    txt = "\n".join(lines)
    print(txt)
    (B.OUT / "fixed_noise_analyse.txt").write_text(txt + "\n")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for g in B.MODELS:
        ax[0].plot(psis, [100 * rec[p, g][0] for p in psis], "o-", label=LAB[g])
    ax[0].plot(psis, [100 * np.mean([rec[p, g][0] for g in B.MODELS]) for p in psis], "k--", label="all")
    ax[0].set(xlabel="high-noise auditory sigma (unimodal fit)", ylabel="% recovered (AIC)", ylim=(0, 102), title="model recovery")
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].plot(psis, [pc_err[p] for p in psis], "o-", color="tab:blue")
    ax[1].set(xlabel="high-noise auditory sigma", ylabel="RMSE", title="p_c (causal inference)", ylim=(0, None))
    ax[2].plot(psis, [ps_err[p] for p in psis], "o-", color="tab:green")
    ax[2].set(xlabel="high-noise auditory sigma", ylabel="RMSE", title="p_switch at high noise", ylim=(0, None))
    for a in ax:
        a.set_xscale("log"); a.set_xticks(psis); a.set_xticklabels([f"{p:g}" for p in psis])
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = HERE / "report" / "fixed_noise"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=130)


if __name__ == "__main__":
    main()
