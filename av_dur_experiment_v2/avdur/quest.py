"""QUEST+ (psi-marginal) for the unimodal blocks.

The participant's psychometric function is P(test judged longer) = lapse/2 + (1 - lapse) *
Phi((ln(test/standard) - mu) / sigma), the screening's model. QUEST+ keeps a probability for
every (mu, sigma, lapse) on a grid. Before each trial it chooses the test duration whose answer
is expected to leave the least uncertainty (entropy) about mu and sigma; the lapse is estimated
but not targeted (Prins 2013). After the answer it updates the probabilities with Bayes' rule.
Kontsevich & Tyler 1999; Watson 2017. analysis/quest_unimodal_simulation.py compares it with the
four staircases it replaces.

QuestStair wraps it in the staircase interface the session uses (.level, .update), so the trial
loop, the order balancing, the breaks and the screening fit are unchanged. Candidate tests are
whole frames, so the update uses exactly the duration shown.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

import experiment_settings as S


class QuestPlus:
    def __init__(self, x_grid, mu_grid, sigma_grid, lapse_grid, target="mu_sigma"):
        self.x = np.asarray(x_grid, float)
        self.mu, self.sigma, self.lapse = (np.asarray(g, float) for g in (mu_grid, sigma_grid, lapse_grid))
        m, s, l = np.meshgrid(self.mu, self.sigma, self.lapse, indexing="ij")
        self.p_mu, self.p_sigma, self.p_lapse = m.ravel(), s.ravel(), l.ravel()
        self.table = self.p_lapse / 2 + (1 - self.p_lapse) * norm.cdf(
            (self.x[:, None] - self.p_mu[None, :]) / self.p_sigma[None, :])          # (n_x, n_theta)
        self.post = np.full(self.p_mu.size, 1.0 / self.p_mu.size)                    # flat prior
        self.target = target

    def _entropy(self, joint):
        if self.target == "mu":
            marg = joint.reshape(joint.shape[0], self.mu.size, -1).sum(-1)
        else:                                                    # (mu, sigma); lapse summed out
            marg = joint.reshape(joint.shape[0], self.mu.size * self.sigma.size, self.lapse.size).sum(-1)
        marg = marg / marg.sum(1, keepdims=True)
        return -np.sum(marg * np.log(np.where(marg > 0, marg, 1.0)), axis=1)

    def next_index(self):
        p1 = self.table @ self.post
        expected = p1 * self._entropy(self.table * self.post) + (1 - p1) * self._entropy((1 - self.table) * self.post)
        return int(np.argmin(expected))

    def update(self, x, chose_test):
        p = self.p_lapse / 2 + (1 - self.p_lapse) * norm.cdf((x - self.p_mu) / self.p_sigma)
        self.post = self.post * (p if chose_test else 1 - p)
        self.post /= self.post.sum()

    def estimate(self):
        """Posterior means: mu, sigma (geometric) and lapse."""
        return (float(self.post @ self.p_mu), float(np.exp(self.post @ np.log(self.p_sigma))),
                float(self.post @ self.p_lapse))


def grids(cfg=None):
    q = cfg or S.QUEST
    return dict(mu_grid=np.linspace(*q["mu_range"], q["n_mu"]),
                sigma_grid=np.exp(np.linspace(np.log(q["sigma_range"][0]), np.log(q["sigma_range"][1]), q["n_sigma"])),
                lapse_grid=np.linspace(0.0, q["lapse_max"], q["n_lapse"]))


class QuestStair:
    """One condition's QUEST+, with the staircase interface (level, update, start, reversals)."""

    rule = "QUEST"
    start = "QUEST"
    reversals = ""
    is_reversal = ""

    def __init__(self, floor, ceiling, standard_s, frame_rate, cfg=None):
        q = cfg or S.QUEST
        std = int(round(standard_s * frame_rate))
        lo = max(1, math.ceil(std * (1 + floor) - 1e-9))
        hi = int(math.floor(std * (1 + ceiling) + 1e-9))
        frames = np.unique(np.round(np.geomspace(lo, hi, q["n_levels"])).astype(int))
        self.levels = frames / std - 1                       # test/standard - 1, as the staircases
        self.q = QuestPlus(np.log(frames / std), target=q.get("target", "mu_sigma"), **grids(q))
        self._pending = None

    @property
    def level(self):
        if self._pending is None:
            self._pending = self.q.next_index()
        return float(self.levels[self._pending])

    def update(self, chose_test):
        """Answer to the trial at .level."""
        if self._pending is None:
            raise RuntimeError("QuestStair.update before .level was read")
        self.q.update(self.q.x[self._pending], chose_test)
        self._pending = None

    def observe(self, level, chose_test):
        """An answer at another level (an easy trial of the same condition): informs the lapse."""
        self.q.update(math.log(1 + level), chose_test)
        self._pending = None                                 # the best next test may have changed

    def estimate(self):
        return self.q.estimate()
