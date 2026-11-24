"""Wall-clock time of experiment 09, end to end, from file modification times and the arm logs
(no timing log was kept for the extraction). Run: python tmp/exp09/wallclock.py"""
import datetime as dt
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TRACKS, FEATS = Path("outputs/cache/tracks/2a11240a2d42"), Path("outputs/cache/features/4c761244568f")
OUT = Path("tmp/exp09")
ts = lambda p: dt.datetime.fromtimestamp(p.stat().st_mtime)
fmt = lambda t: t.strftime("%d %b %H:%M")
hm = lambda s: f"{s/3600:.1f} h" if s >= 5400 else f"{s/60:.0f} min"

# ---- extraction: frame pass (.npz) and feature pass (.csv, written last per recording)
rec = {(r["group"], r["experiment"], r["camera"]): r for r in json.loads(Path("outputs/dataset/recordings.json").read_text())}
def key(name):
    g, e, c = name.split(".")[0].split("__")
    return g, e, c
def layer(name):
    m = re.search(r"\.aug(\d+)-", name)
    return int(m.group(1)) if m else 0
npz = sorted(TRACKS.glob("*.npz"), key=ts)
csv = sorted((p for p in FEATS.glob("*.csv") if not p.name.endswith((".labelwin.csv", ".tracks.csv"))), key=ts)
print(f"Extraction: {len(npz)} frame-pass files, {len(csv)} window files (the window .csv is written last)")
t_csv = sorted(ts(p) for p in csv)
first, last = t_csv[0], t_csv[-1]
gaps = np.diff([t.timestamp() for t in t_csv])
big = [(t_csv[i], g) for i, g in enumerate(gaps) if g > 3600]
print(f"  first recording finished {fmt(first)}, last {fmt(last)}; span {hm((last-first).total_seconds())}")
for t, g in big:
    print(f"  pause after {fmt(t)}: {hm(g)} with no recording finished (run stopped and resumed)")
active = (last - first).total_seconds() - sum(g for _, g in big)
for l in sorted({layer(p.name) for p in csv}):
    t = sorted(ts(p) for p in csv if layer(p.name) == l)
    print(f"  {'originals' if l == 0 else f'copy {l}'}: {len(t)} recordings, finished {fmt(t[0])} - {fmt(t[-1])}")

# per-stage CPU/GPU seconds per recording, from each recording's manifest (stage_seconds)
stages, n_frames, video_s, per_rec = defaultdict(float), 0, 0.0, []
for p in csv:
    m = json.loads(p.with_name(p.name[:-4] + ".manifest.json").read_text(encoding="utf-8"))
    if m.get("frame_pass_cached"):
        continue
    for k, v in m["stage_seconds"].items():
        stages[k] += v
    n_frames += m["timing"]["n_frames"]; video_s += m["timing"]["duration_s"]
    per_rec.append(sum(m["stage_seconds"].values()))
work = sum(stages.values())
print(f"  {len(per_rec)} recordings, {video_s/3600:.2f} h of video, {n_frames:,} frames")
print(f"  active wall time (span minus pauses, plus about one recording before the first finished):"
      f" {hm(active + np.median(per_rec))}")
print(f"  summed per-recording time {hm(work)} -> {work/(active + np.median(per_rec)):.2f} recordings in parallel on average")
print(f"  per recording: median {hm(np.median(per_rec))}, range {hm(min(per_rec))} - {hm(max(per_rec))}")
print(f"  wall throughput: {n_frames/(active + np.median(per_rec)):.1f} frames/s, {video_s/(active + np.median(per_rec)):.2f}x real time")
print("  | Stage | Total | ms per frame | Share |")
print("  |---|---|---|---|")
for k, v in sorted(stages.items(), key=lambda kv: -kv[1]):
    print(f"  | {k} | {hm(v)} | {v/n_frames*1000:.1f} | {v/work:.0%} |")
print(f"  | total | {hm(work)} | {work/n_frames*1000:.1f} | |")

# ---- label-window cache (built once by the first training run)
lw = sorted(FEATS.glob("*.labelwin*"), key=ts)
if lw:
    a, b = ts(lw[0]), ts(lw[-1])
    print(f"\nLabel-window cache: {len(lw)} files, {fmt(a)} - {fmt(b)} ({hm((b-a).total_seconds())})")

# ---- training arms: start from arms.log (HH:MM), end = the arm log's mtime
starts = {}
for line in Path(OUT / "arms.log").read_text(encoding="utf-8").splitlines():
    m = re.match(r"=== (\S+) \((\d\d):(\d\d)\)", line)
    if m:
        starts[m.group(1)] = (int(m.group(2)), int(m.group(3)))      # the last attempt wins
rows = []
for name, (h, mi) in starts.items():
    log = OUT / f"{name}.log"
    if not log.exists() or "config" not in log.read_text(encoding="utf-8", errors="ignore"):
        continue
    end = ts(log)
    start = end.replace(hour=h, minute=mi, second=0, microsecond=0)
    if start > end:
        start -= dt.timedelta(days=1)
    rows.append((name, start, end, (end - start).total_seconds()))
model = lambda n: n.split("_")[0]
comp = lambda n: n.split("_")[1]
groups = defaultdict(list)
for n, s, e, d in rows:
    groups[(model(n), "c3" if comp(n).startswith("c3") else comp(n))].append(d)
names = {"lr": "logistic regression", "rf": "random forest", "knn": "k-NN classifier, k=500",
         "knn5": "k-NN classifier, k=5", "knnr5": "k-NN regression, k=5", "knnrk15": "k-NN regression, k=15",
         "knnrk50": "k-NN regression, k=50"}
print(f"\nTraining: {len(rows)} finished arms with a start time (min per arm: median [min - max])")
print(f"| Model | Comparison | Arms | Median | Range | Total |\n|---|---|---|---|---|---|")
for (m, c) in sorted(groups, key=lambda k: (list(names).index(k[0]) if k[0] in names else 99, k[1])):
    d = np.array(groups[(m, c)]) / 60
    print(f"| {names.get(m, m)} | {c} | {len(d)} | {np.median(d):.1f} | {d.min():.1f} - {d.max():.1f} | {hm(d.sum()*60)} |")
tot = sum(r[3] for r in rows)
print(f"Sum over arms: {hm(tot)}")
