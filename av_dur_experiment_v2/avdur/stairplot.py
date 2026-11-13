"""Staircase plot of one session: every trial's test duration in session order, one line per
staircase, one panel per condition (also used by analysis/04_staircases.ipynb)."""

from __future__ import annotations

import numpy as np

from avdur import psychometrics as pm

STAIR_COLORS = {'2D1U': 'tab:red', '1D2U': 'tab:blue', '3D1U': 'tab:orange', '1D3U': 'tab:cyan',
                '1D1U': 'tab:purple', '1D1U_b': 'tab:pink', 'QUEST': 'tab:green'}
TRUE = ['1', 'True', 'true', True, 1]


def condition_label(task, noise, conflict, standard_ms=None):
    """standard_ms: given when the session interleaves several standards (cross-modal)."""
    if task == 'unimodal_visual':
        return 'visual'
    label = (f'standard {standard_ms:.0f} ms, ' if standard_ms is not None else '') + f'noise {noise:g}'
    return label + (f', conflict {1000 * conflict:+.0f} ms' if task == 'main' else '')


def y_range(d, std_ms):
    """Y limits (ms) that hold every staircase trial and the standard, with a small margin."""
    s = d.test_ms[d.kind == 'staircase']
    lo, hi = min(s.min(), std_ms) if len(s) else std_ms, max(s.max(), std_ms) if len(s) else std_ms
    pad = 0.06 * (hi - lo) + 10
    return max(lo - pad, 0), hi + pad


def plot_staircases(d, title='', show_fit=True):
    """Test duration (ms) of every trial in session order, one line per staircase, one panel per
    condition. Filled marker: test judged longer; open: standard judged longer; black ring:
    reversal; grey x: easy (lapse) trial, grey triangle on the panel edge: easy trial outside the
    staircases' range. Dotted line: standard; dashed: fitted PSE."""
    task = d.task.iloc[0]
    d = d[d.kind != 'warmup'].sort_values('trial_num').copy()       # warm-up trials are not analysed
    d['test_ms'] = 1000 * d.testDurS
    d['noise'] = d.audNoise.fillna(-1)
    d['std_ms'] = (1000 * d.standardDur).round()
    several = d.std_ms.nunique() > 1                        # cross-modal standards interleaved: one row per standard
    keys = sorted(d.groupby(['std_ms', 'noise', 'conflict_nominal']).groups)
    if task == 'main':
        n_cols = len({k[2] for k in keys})
    elif several:
        n_cols = len({k[1] for k in keys})
    else:
        n_cols = len(keys)
    n_rows = len(keys) // n_cols
    width = 11 if n_cols == 1 else (8 if n_cols == 2 else 5.5)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(min(width * n_cols, 26), 5 * n_rows), squeeze=False,
                             sharey=not several, sharex=True)
    for ax, (std_ms, noise, conflict) in zip(axes.flat, keys):
        c = d[(d.std_ms == std_ms) & (d.noise == noise) & (d.conflict_nominal == conflict)]
        name = condition_label(task, None if noise < 0 else noise, conflict, std_ms if several else None)
        for stair, s in c[c.kind == 'staircase'].groupby('current_stair'):
            col = STAIR_COLORS.get(stair, 'black')
            start = s.stair_start.iloc[0] if 'stair_start' in s else ''
            ax.plot(s.trial_num, s.test_ms, '-', color=col, lw=1, alpha=0.7,
                    label='QUEST+' if stair == 'QUEST' else f'{stair} (from {start})')
            yes = s.chose_test == 1
            ax.scatter(s.trial_num[yes], s.test_ms[yes], s=22, color=col, zorder=3)
            ax.scatter(s.trial_num[~yes], s.test_ms[~yes], s=22, facecolors='white', edgecolors=col, zorder=3)
            rev = s.stair_is_reversal.isin(TRUE)
            ax.scatter(s.trial_num[rev], s.test_ms[rev], s=70, facecolors='none', edgecolors='black', lw=1, zorder=4)
        # The y range follows the staircases, not the easy trials (from 13 Nov 2026): an easy trial outside it
        # (e.g. 2000 ms) is drawn on the panel edge as a grey triangle, so the staircases keep the room.
        std = 1000 * c.standardDur.iloc[0]
        lo, hi = y_range(d if not several else c, std)
        ax.set_ylim(lo, hi)
        lapse = c[c.kind == 'lapse']
        inside = lapse.test_ms.between(lo, hi)
        ax.scatter(lapse.trial_num[inside], lapse.test_ms[inside], marker='x', color='0.5', s=25, label='easy trials')
        for side, y, mk in (('above', hi, '^'), ('below', lo, 'v')):
            out = lapse[lapse.test_ms > hi] if side == 'above' else lapse[lapse.test_ms < lo]
            if len(out):
                ms = ', '.join(f'{v:.0f}' for v in sorted(out.test_ms.round().unique()))
                ax.scatter(out.trial_num, np.full(len(out), y), marker=mk, color='0.5', s=22, clip_on=False,
                           zorder=5, label=f'easy trials at {ms} ms (drawn on the edge)')
        ax.axhline(std, color='0.4', ls=':', lw=1)
        if show_fit and c.responses.isin([1, 2]).sum() > 20:
            p = pm.fit(c.assign(cnd='x'), 'cnd', 'lognormal', lapse_max=0.1)['params'].iloc[0]
            ax.axhline(std * np.exp(p.mu), color='black', ls='--', lw=1)
            ax.set_title(f'{name}\nPSE {std * np.exp(p.mu):.0f} ms, σ {p.sigma:.2f}', fontsize=11)
        else:
            ax.set_title(name, fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel('Trial number in the session')
    for ax in axes[:, 0]:
        ax.set_ylabel('Test duration (ms)')
    handles, labels = [], []
    for ax in axes.flat:                                 # every rule once (panels can have different rules)
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    handles += [Line2D([], [], marker='o', color='0.3', ls=''), Line2D([], [], marker='o', mfc='white', color='0.3', ls=''),
                Line2D([], [], marker='o', mfc='none', mec='black', ms=9, ls=''),
                Line2D([], [], color='0.4', ls=':'), Line2D([], [], color='black', ls='--')]
    labels += ['test judged longer', 'standard judged longer', 'reversal', 'standard', 'fitted PSE']
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=10)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def save(d, title, path):
    import matplotlib.pyplot as plt
    fig = plot_staircases(d, title)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return path
