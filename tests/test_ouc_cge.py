"""OUC-CGE adapters: split parsing, the nested video lookup, and the video frame source."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.datasets.ouc_cge import OucCgeIndex, find_videos, read_splits
from src.ingestion import FRAME_SOURCES


def write_video(path: Path, n: int, fps: float, wh=(64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, wh)
    for i in range(n):
        img = np.full((wh[1], wh[0], 3), i % 255, np.uint8)
        w.write(img)
    w.release()


class TestSplits(unittest.TestCase):
    def test_parse_skips_placeholder_and_checks_labels(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "train.csv").write_text("videos/high/view1.mp4 2\nvideos/low/view1.mp4 0\n")
            Path(d, "val.csv").write_text("videos/mid/view3.mp4 1\n\n")
            Path(d, "test.csv").write_text("videos/low/view9.mp4 0\nvideos/anonymity/a.mp4\n")
            clips = read_splits(d, verbose=False)
            self.assertEqual([c.clip_id for c in clips], ["high_view1", "low_view1", "mid_view3", "low_view9"])
            self.assertEqual([c.split for c in clips], ["train", "train", "val", "test"])
            self.assertEqual(clips[0].key.camera, "view1")
            Path(d, "val.csv").write_text("videos/mid/view3.mp4 2\n")           # folder says mid, label says high
            with self.assertRaises(ValueError):
                read_splits(d, verbose=False)


class TestVideoLookup(unittest.TestCase):
    def test_nested_case_insensitive_and_samples_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            for rel in ["high/high-2(1-2)/view1.mp4", "Low/low-3(1-1000)/low-3(1-1000)/view7.mp4",
                        "Low/low-3(1-1000)/low-3(1-1000)/7.mp4", "mid/mid sample/view1.mp4", "mid/mid-1/view2.mp4"]:
                Path(d, rel).parent.mkdir(parents=True, exist_ok=True)
                Path(d, rel).write_bytes(b"x")
            self.assertEqual(sorted(find_videos(d)), ["high_view1", "low_view7", "mid_view2"])

    def test_index_reports_missing(self):
        with tempfile.TemporaryDirectory() as d:
            for s, txt in [("train", "videos/high/view1.mp4 2\n"), ("val", "videos/mid/view2.mp4 1\n"),
                           ("test", "videos/low/view5.mp4 0\n")]:
                Path(d, f"{s}.csv").write_text(txt)
            Path(d, "vids/high/a").mkdir(parents=True)
            Path(d, "vids/high/a/view1.mp4").write_bytes(b"x")
            idx = OucCgeIndex(Path(d, "vids"), d)
            self.assertEqual([c.clip_id for c in idx.clips()], ["high_view1"])
            self.assertEqual(sorted(c.clip_id for c in idx.missing()), ["low_view5", "mid_view2"])


class TestVideoFrameSource(unittest.TestCase):
    def test_target_fps_stride_and_cap(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "v.mp4")
            write_video(p, 60, 30.0)
            src = FRAME_SOURCES.build("video", path=p, target_fps=5)
            frames = list(src)
            self.assertEqual(src.step, 6)
            self.assertEqual(len(frames), 10)
            self.assertTrue(np.allclose([f.t for f in frames], np.arange(10) * 0.2))
            self.assertEqual([f.index for f in frames], list(range(10)))
            self.assertEqual(frames[0].image.shape, (48, 64, 3))
            self.assertEqual(len(list(FRAME_SOURCES.build("video", path=p, target_fps=5, max_frames=3))), 3)
            self.assertEqual(len(list(FRAME_SOURCES.build("video", path=p, stride=2))), 30)

    def test_unopenable_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OSError):
                FRAME_SOURCES.build("video", path=Path(d, "missing.mp4"))


if __name__ == "__main__":
    unittest.main()
