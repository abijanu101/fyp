"""scripts/clustering/behaviour_profile.py on synthetic data with a known answer: a feature that
follows the ratings only within sessions, one whose student average follows the students' levels
only, and features that follow nothing."""

import unittest

import numpy as np
import pandas as pd

from scripts.clustering.behaviour_profile import between_profile, entry_table, within_profile


def fake(seed: int, n_cls: int = 18, n_stu: int = 15, n_t: int = 20):
    """Entries every few seconds for 2 experts; per student a level and a mean behaviour; features:
      f_within   varies within the session with the rating (its student mean is noise)
      f_between  its student mean follows the student's level (constant within the session + noise)
      f_null     unrelated"""
    rng = np.random.default_rng(seed)
    ent, lev, prof = [], [], []
    for c in range(n_cls):
        cls = f"group_0{c % 3 + 1}/experiment_0{c // 3 + 1}"
        for s in range(n_stu):
            sid = f"subject_{s + 1:02d}"
            lvl = rng.normal(0, 0.5)
            prof.append({"cls": cls, "sid": sid, "f_between": lvl + rng.normal(0, 0.3), "f_within": rng.normal(),
                         "f_null": rng.normal()})
            for lab in ("labeler_01", "labeler_02"):
                lev.append({"cls": cls, "sid": sid, "labeler": lab, "level": 3 + lvl + rng.normal(0, 0.3)})
                for t in range(n_t):
                    w = rng.normal()
                    ent.append({"entry_id": f"{cls}/{sid}/{lab}/{t}", "cls": cls, "sess": f"{cls}/{sid}", "labeler": lab,
                                "rating": float(np.clip(np.round(3 + lvl + 0.8 * w + rng.normal(0, 0.5)), 1, 5)),
                                "f_within": w + rng.normal(0, 0.3), "f_between": lvl + rng.normal(0, 1.0),
                                "f_null": rng.normal()})
    return pd.DataFrame(ent), pd.DataFrame(lev), pd.DataFrame(prof)


COLS = ["f_within", "f_between", "f_null"]


class TestBehaviourProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        e, lv, prof = fake(1)
        rng = np.random.default_rng(0)
        cls.w = within_profile(e, COLS, 200, rng)
        cls.b = between_profile(lv, prof, COLS, 300, 200, rng)

    def test_within_feature_found_within_only(self):
        self.assertGreater(self.w["f_within"]["spearman"], 0.4)
        self.assertLess(self.w["f_within"]["p_boot"], 0.01)
        self.assertLess(abs(self.b["f_within"]["spearman"]), 0.1)

    def test_between_feature_found_between_only(self):
        self.assertGreater(self.b["f_between"]["spearman"], 0.4)
        self.assertLess(self.b["f_between"]["p_perm"], 0.01)
        # within a session f_between is the student's level plus noise: centred within the session,
        # only the noise is left, unrelated to the rating's within-session part
        self.assertLess(abs(self.w["f_between"]["spearman"]), 0.05)

    def test_false_positive_rates(self):
        pw, pb = [], []
        for s in range(30):
            e, lv, prof = fake(100 + s, n_cls=18, n_stu=10, n_t=10)
            rng = np.random.default_rng(s)
            pw.append(within_profile(e, ["f_null"], 200, rng)["f_null"]["p_boot"])
            pb.append(between_profile(lv, prof, ["f_null"], 200, 0, rng)["f_null"]["p_perm"])
        self.assertLessEqual(np.mean(np.array(pw) < 0.05), 0.15, pw)
        self.assertLessEqual(np.mean(np.array(pb) < 0.05), 0.15, pb)

    def test_entry_table_averages_an_entrys_cameras(self):
        r = pd.DataFrame({"entry_id": ["a", "a", "b"], "rating": [3.0, 3.0, 4.0], "labeler": "labeler_01",
                          "sess": "s", "cls": "c", "x": [1.0, 3.0, 5.0]})
        e = entry_table(r, ["x"]).set_index("entry_id")
        self.assertEqual(e.loc["a", "x"], 2.0)
        self.assertEqual(len(e), 2)


if __name__ == "__main__":
    unittest.main()
