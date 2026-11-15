"""Per-labeler ensemble (config engagement.labelers = "ensemble"; src/experts/levels.py)."""

import unittest

import numpy as np

from src.experts import EngagementRows, ExpertMixture, LabelerEnsemble, fit_engagement, predict_windows
from src.experts.expert import SklearnExpert

CLASSES = ["01", "02"]
ENSEMBLE = {"labelers": "ensemble", "ensemble_exclude": ["self_labeling"]}


def mixture():
    return ExpertMixture(CLASSES, lambda exp_id: SklearnExpert("logreg"), "shared", "soft")


class TestLabelerEnsemble(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 400
        self.x = rng.normal(size=n)
        self.window = np.repeat(np.arange(n // 4), 4)               # 100 windows, 4 students each
        a = (self.x > 0).astype(float)                             # labeler A: engaged when x > 0
        b = 1 - a                                                  # labeler B: the opposite
        self.labels = {"labeler_01": a, "labeler_02": b, "self_labeling": a}
        self.tr, self.te = np.arange(50), np.arange(50, 100)
        self.g = np.tile([1.0, 0.0], (len(self.te), 1))
        self.share = np.array([(self.x[self.window == w] > 0).mean() for w in self.te])

    def rows(self, rated_b: bool | None):
        n = len(self.x)
        rated = None if rated_b is None else {"labeler_01": np.ones(n, bool), "labeler_02": np.full(n, rated_b),
                                              "self_labeling": np.ones(n, bool)}
        return EngagementRows(self.x[:, None], ["trk.x"], self.window, np.full(n, "01"), self.labels["labeler_01"],
                              y_by_labeler=self.labels, rated_by=rated)

    def score(self, rows):
        model, fill = fit_engagement(ENSEMBLE, mixture, rows, self.tr)
        self.assertIsInstance(model, LabelerEnsemble)
        self.assertEqual(sorted(model.mixtures), ["labeler_01", "labeler_02"])   # self-ratings excluded
        return np.corrcoef(predict_windows(model, rows, self.te, self.g, fill), self.share)[0, 1]

    def test_a_labeler_who_did_not_rate_a_student_does_not_vote(self):
        self.assertGreater(self.score(self.rows(rated_b=False)), 0.8)         # only A rated: A's view

    def test_both_labelers_vote_where_both_rated(self):
        self.assertLess(abs(self.score(self.rows(rated_b=True))), 0.5)        # opposite views cancel

    def test_without_labeler_information_every_labeler_votes(self):
        self.assertLess(abs(self.score(self.rows(rated_b=None))), 0.5)        # inference: average all

    def test_consensus_mode_fits_one_mixture(self):
        model, _ = fit_engagement({"labelers": "consensus"}, mixture, self.rows(None), self.tr)
        self.assertIsInstance(model, ExpertMixture)


if __name__ == "__main__":
    unittest.main()
