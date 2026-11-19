"""Derived config constants (src/core/derive.py, configs/derived.py)."""

import json
import tempfile
import unittest
from pathlib import Path

from src.core.config import config_hash, load_config
from src.core.derive import Sourced, derived_values, result


class TestDerive(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = Path(self.dir.name)
        (d / "res.json").write_text(json.dumps({"optimal_threshold": 0.11,
                                                "config_patch": {"tracker.options": {"a": 1}}}))
        (d / "derived.py").write_text(
            "from src.core.derive import result\n"
            f"R = r'{(d / 'res.json').as_posix()}'\n"
            "OVERRIDES = {'detector.conf': result(R, 'optimal_threshold'),\n"
            "             'tracker.options': result(R, 'config_patch.tracker.options'),\n"
            "             'window.length_s': 6}\n")
        base = json.loads(Path("configs/deliverable3.json").read_text(encoding="utf-8"))
        base["derived"] = (d / "derived.py").as_posix()
        self.cfg_path = d / "cfg.json"
        self.cfg_path.write_text(json.dumps(base))
        self.d = d

    def tearDown(self):
        self.dir.cleanup()

    def test_values_sources_and_keys_with_dots(self):
        r = result(self.d / "res.json", "config_patch.tracker.options")
        self.assertEqual(r, Sourced({"a": 1}, f"{(self.d / 'res.json').as_posix()}#config_patch.tracker.options"))
        values, sources = derived_values(self.d / "derived.py")
        self.assertEqual(values, {"detector.conf": 0.11, "tracker.options": {"a": 1}, "window.length_s": 6})
        self.assertTrue(sources["window.length_s"].endswith("derived.py"))   # plain values: the module

    def test_load_config_applies_them_before_set_and_hashes_values_only(self):
        cfg = load_config(self.cfg_path)
        self.assertEqual((cfg["detector"]["conf"], cfg["tracker"]["options"]), (0.11, {"a": 1}))
        self.assertEqual(load_config(self.cfg_path, {"detector.conf": 0.3})["detector"]["conf"], 0.3)
        moved = dict(cfg, derived_from={"detector.conf": "elsewhere"}, derived="other.py")
        self.assertEqual(config_hash(cfg), config_hash(moved))              # the source is not hashed

    def test_parameters_from_the_config(self):
        """`derived` as {"module", **params}: the module's overrides(params) reads the files the config names."""
        (self.d / "param.py").write_text(
            "from src.core.derive import result\n"
            "def overrides(params):\n"
            "    return {'detector.conf': result(params['scan'], 'optimal_threshold')}\n")
        values, sources = derived_values({"module": (self.d / "param.py").as_posix(),
                                          "scan": (self.d / "res.json").as_posix()})
        self.assertEqual(values, {"detector.conf": 0.11})
        self.assertTrue(sources["detector.conf"].endswith("res.json#optimal_threshold"))
        with self.assertRaises(TypeError):                  # a fixed OVERRIDES module takes no parameters
            derived_values({"module": (self.d / "derived.py").as_posix(), "scan": "x"})

    def test_the_project_configs_load(self):
        """configs/derived.py with each config's own scans: the base (D2/T2) and experiment 09 (D3/T3)."""
        base = load_config("configs/deliverable3.json")
        vfoa = load_config("docs/experiments/09_vfoa.json")
        self.assertEqual(base["tracker"]["options"]["det_thresh"], base["detector"]["conf"])
        self.assertEqual(vfoa["tracker"]["options"]["det_thresh"], vfoa["detector"]["conf"])
        self.assertNotEqual(base["detector"]["conf"], vfoa["detector"]["conf"])
        self.assertIn("_pose", vfoa["derived_from"]["tracker.options"])

    def test_a_missing_result_fails_loudly(self):
        with self.assertRaises(FileNotFoundError):
            result(self.d / "missing.json", "x")
        with self.assertRaises(KeyError):
            result(self.d / "res.json", "config_patch.nope")


if __name__ == "__main__":
    unittest.main()
