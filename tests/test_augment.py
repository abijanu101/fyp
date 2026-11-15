"""Session-level data augmentation (src/ingestion/augment.py, config `augmentation`)."""

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.core.config import load_config
from src.ingestion import AugmentedZipFrameSource, ZipFrameSource, aug_tag, hflip_boxes, sample_augmentation
from src.ingestion.augment import augment_frame

AUG = load_config("configs/deliverable3.json")["augmentation"]


def fake_zip(path: Path, n: int = 4, w: int = 64, h: int = 48, smooth: bool = False) -> Path:
    """n PNG frames of random noise, or (smooth) of a horizontal gradient: JPEG-friendly, flip-visible."""
    rng = np.random.default_rng(0)
    ramp = np.broadcast_to(np.linspace(0, 255, w, dtype=np.uint8)[None, :, None], (h, w, 3)).copy()
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n):
            img = ramp if smooth else rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            ok, buf = cv2.imencode(".png", img)
            zf.writestr(f"images/10_00_0{i}_000000.png", buf.tobytes())
    return path


class TestAugmentation(unittest.TestCase):
    def test_hflip_mirrors_boxes(self):
        b = np.array([[10.0, 5.0, 30.0, 20.0]])
        np.testing.assert_array_equal(hflip_boxes(b, 100), [[70.0, 5.0, 90.0, 20.0]])
        np.testing.assert_array_equal(hflip_boxes(hflip_boxes(b, 100), 100), b)

    def test_parameters_are_fixed_per_session_and_copy(self):
        a = sample_augmentation(AUG, "group_01/experiment_01/context_cam_01", 1)
        self.assertEqual(a, sample_augmentation(AUG, "group_01/experiment_01/context_cam_01", 1))
        self.assertNotEqual(a, sample_augmentation(AUG, "group_01/experiment_01/context_cam_01", 2))
        self.assertNotEqual(a, sample_augmentation(AUG, "group_02/experiment_01/context_cam_01", 1))

    def test_hue_shift_stays_small(self):
        hues = [sample_augmentation(AUG, f"s{i}", c)["hue_deg"] for i in range(50) for c in (1, 2)]
        self.assertLessEqual(max(abs(h) for h in hues), AUG["hue_deg"])
        self.assertLessEqual(AUG["hue_deg"], 5)

    def test_tag_ignores_copy_count(self):
        self.assertEqual(aug_tag({**AUG, "copies": 1}, 1), aug_tag({**AUG, "copies": 3}, 1))
        self.assertNotEqual(aug_tag(AUG, 1), aug_tag({**AUG, "hue_deg": 4}, 1))

    def test_same_transform_on_every_frame_noise_aside(self):
        p = {**sample_augmentation(AUG, "s", 1), "noise": 0.0, "hflip": True}
        frames = [np.full((8, 8, 3), v, np.uint8) for v in (40, 40)]
        a, b = (augment_frame(f, p, i) for i, f in enumerate(frames))
        np.testing.assert_array_equal(a, b)          # identical input -> identical output across frames

    def test_source_is_a_dataset_with_the_original_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            z = fake_zip(Path(d) / "cam.zip")
            p = {**sample_augmentation(AUG, "s", 1), "hflip": True, "noise": 0.0, "mask_kind": "none",
                 "mask_rects": [], "brightness": 0.0, "contrast": 1.0, "saturation": 1.0, "hue_deg": 0.0,
                 "gamma": 1.0}
            src, orig = AugmentedZipFrameSource(z, p), ZipFrameSource(z)
            self.assertIsInstance(src, Dataset)
            self.assertEqual(len(src), 4)
            np.testing.assert_array_equal(src.timestamps, orig.timestamps)
            np.testing.assert_array_equal(src[2].image, orig.read(2).image[:, ::-1])   # pure flip
            self.assertEqual([f.index for f in src], [0, 1, 2, 3])

    def test_a_copy_is_the_same_video_every_time_it_is_read(self):
        # Copies are never written to disk: seeding alone must make every read identical, noise included.
        with tempfile.TemporaryDirectory() as d:
            z = fake_zip(Path(d) / "cam.zip")
            p = {**sample_augmentation(AUG, "group_01/experiment_01/context_cam_01", 1), "noise": 0.3}
            first = [f.image for f in AugmentedZipFrameSource(z, p)]
            again = [f.image for f in AugmentedZipFrameSource(z, sample_augmentation(AUG, "group_01/experiment_01/context_cam_01", 1)
                                                              | {"noise": 0.3})]
            for a, b in zip(first, again):
                np.testing.assert_array_equal(a, b)
            other = AugmentedZipFrameSource(z, {**sample_augmentation(AUG, "group_01/experiment_01/context_cam_01", 2),
                                                "noise": 0.3})[0].image
            self.assertFalse(np.array_equal(first[0], other))  # another copy is a different video

    def test_augmented_rows_never_reach_a_test_fold(self):
        from scripts.train_models import heldout_splits
        df = pd.DataFrame({"group": ["g1", "g1", "g2", "g2", "g1", "g2"],
                           "aug": [False, False, False, False, True, True]})
        for g, tr, te in heldout_splits(df):
            self.assertFalse(df["aug"].to_numpy()[te].any())
            self.assertTrue(df["aug"].to_numpy()[tr].any())          # copies of the other group train
            self.assertTrue((df["group"].to_numpy()[tr] != g).all())


if __name__ == "__main__":
    unittest.main()
