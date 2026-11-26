"""Compare association.method per_frame vs mean_box on the cached tracks of a config's originals."""
import sys
from pathlib import Path
from src.core.config import load_config
from src.core.store import load_track
from src.datasets import DipserIndex
from src.tracking import associate_sids
from scripts.run_pipeline import tracks_dir, session_stem

cfg = load_config(sys.argv[1]); a = cfg["association"]
index = DipserIndex.from_config(cfg)
n = same = 0; diff_sessions = []
for key in index.sessions():
    cands = [tracks_dir(cfg) / f"{session_stem(key)}.npz"]
    if not cands[0].exists(): continue
    tr = load_track(cands[0]).tracklets
    subj = index.annotations(key)
    pf = associate_sids(tr, subj, a["min_iou"], a["min_match_share"], "per_frame")
    mb = associate_sids(tr, subj, a["min_iou"], a["min_match_share"], "mean_box")
    n += 1; same += pf == mb
    if pf != mb:
        only_pf = {t: s for t, s in pf.items() if mb.get(t) != s}; only_mb = {t: s for t, s in mb.items() if pf.get(t) != s}
        diff_sessions.append((cands[0].stem, len(pf), len(mb), only_pf, only_mb))
print(f"{same}/{n} sessions identical")
for d in diff_sessions: print(d)
