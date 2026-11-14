import unittest

import numpy as np

from src.experts import EngagementExpert, ExpertMixture, HandWeightedExpert
from src.spu import context_probs, logo_splits

CLASSES = ["01", "02", "03"]


class Constant(EngagementExpert):
    def __init__(self, value):
        self.value = value

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.value)


def mixture(mode, scope="per_experiment"):
    values = {"01": 0.1, "02": 0.5, "03": 0.9, None: 0.3}
    return ExpertMixture(CLASSES, lambda c: Constant(values[c]), scope=scope, mode=mode)


class TestMixture(unittest.TestCase):
    X = np.zeros((2, 3))
    g = np.array([[0.5, 0.5, 0.0], [0.1, 0.2, 0.7]])

    def test_soft(self):
        np.testing.assert_allclose(mixture("soft").predict_proba(self.X, self.g),
                                   [0.5 * 0.1 + 0.5 * 0.5, 0.1 * 0.1 + 0.2 * 0.5 + 0.7 * 0.9])

    def test_hard_routes_to_argmax(self):
        np.testing.assert_allclose(mixture("hard").predict_proba(self.X, self.g), [0.1, 0.9])

    def test_oracle_is_one_hot(self):
        g = context_probs("oracle", CLASSES, np.array(["03", "02"]))
        np.testing.assert_allclose(mixture("soft").predict_proba(self.X, g), [0.9, 0.5])

    def test_shared_scope_uses_one_expert(self):
        np.testing.assert_allclose(mixture("soft", "shared").predict_proba(self.X, self.g), [0.3, 0.3])

    def test_each_expert_trains_on_its_own_experiment(self):
        seen = {}

        class Recorder(Constant):
            def __init__(self, c):
                super().__init__(0.5)
                self.c = c

            def fit(self, X, y):
                seen[self.c] = X[:, 0].tolist()
                return self

        mix = ExpertMixture(CLASSES, Recorder)
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        mix.fit(X, np.array([0, 1, 0, 1]), np.array(["01", "02", "01", "03"]))
        self.assertEqual(seen, {"01": [1.0, 3.0], "02": [2.0], "03": [4.0]})

    def test_hand_weighted_phone_sign(self):
        names = ["vec.still", "vec.phone"]
        X = np.array([[0.0, -1.0], [0.0, 1.0]])
        on_task = HandWeightedExpert(names, {"vec.still": 1.0}, exp_id="01", phone_key="vec.phone",
                                     phone_sign={"01": 1, "02": -1}).fit(X)
        off_task = HandWeightedExpert(names, {"vec.still": 1.0}, exp_id="02", phone_key="vec.phone",
                                      phone_sign={"01": 1, "02": -1}).fit(X)
        self.assertGreater(on_task.predict_proba(X)[1], on_task.predict_proba(X)[0])
        self.assertLess(off_task.predict_proba(X)[1], off_task.predict_proba(X)[0])


class TestLogo(unittest.TestCase):
    def test_held_out_group_is_whole_test_fold(self):
        groups = np.array(["g1", "g1", "g2", "g3", "g2"])
        folds = list(logo_splits(groups))
        self.assertEqual([f[0] for f in folds], ["g1", "g2", "g3"])
        for g, tr, te in folds:
            self.assertTrue((groups[te] == g).all() and (groups[tr] != g).all())


if __name__ == "__main__":
    unittest.main()
