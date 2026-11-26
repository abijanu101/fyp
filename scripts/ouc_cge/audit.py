"""Data audit of OUC-CGE: every split clip found on disk, readable, and its fps / length / size.

  python -m scripts.ouc_cge.audit --root <OUC-CGE videos> --split-dir <OUC-CGE-main> --out <dir>

Writes <out>/clips.csv (one row per clip) and <out>/audit.json (summary). Reads only each file's
header plus its first frame, so it takes minutes, not hours. Needs only opencv, numpy and pandas (no torch): runs in any env, from the repo root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from src.datasets.ouc_cge import find_videos, read_splits


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read()
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"readable": bool(ok), "fps": fps, "n_frames": int(n), "width": w, "height": h,
            "duration_s": n / fps if fps else float("nan"), "bytes": path.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, videos = read_splits(a.split_dir), find_videos(a.root)
    split_ids = {r.clip_id for r in rows}
    records = []
    for i, r in enumerate(rows, 1):
        p = videos.get(r.clip_id)
        rec = {**vars(r), "path": str(p) if p else None, "found": p is not None}
        if p:
            rec |= probe(p)
        records.append(rec)
        if i % 500 == 0:
            print(f"{i}/{len(rows)}", flush=True)
    df = pd.DataFrame(records)
    df.to_csv(out / "clips.csv", index=False)

    ok = df[df["found"] & df["readable"].fillna(False).astype(bool)]
    summary = {
        "split_rows": len(rows),
        "per_split_label": df.groupby(["split", "level"]).size().unstack(fill_value=0).to_dict("index"),
        "found_on_disk": int(df["found"].sum()),
        "missing": df.loc[~df["found"], "clip_id"].tolist(),
        "unreadable": df.loc[df["found"] & ~df["readable"].fillna(False).astype(bool), "clip_id"].tolist(),
        "videos_not_in_splits": sorted(set(videos) - split_ids),
        "fps_counts": ok["fps"].round().astype(int).value_counts().sort_index().to_dict(),
        "resolution_counts": (ok["width"].astype(str) + "x" + ok["height"].astype(str)).value_counts().to_dict(),
        "duration_s": ok["duration_s"].describe(percentiles=[.05, .25, .5, .75, .95]).round(2).to_dict(),
        "duration_by_level_median": ok.groupby("level")["duration_s"].median().round(2).to_dict(),
        "fps_by_level_mean": ok.groupby("level")["fps"].mean().round(2).to_dict(),
        "share_shorter_than_9s": round(float((ok["duration_s"] < 9).mean()), 4),
        "total_gb": round(float(ok["bytes"].sum()) / 1e9, 2),
    }
    (out / "audit.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("missing", "unreadable")}, indent=2, default=str))
    print(f"missing {len(summary['missing'])}, unreadable {len(summary['unreadable'])}")


if __name__ == "__main__":
    main()
