"""Build one expert model's subsection tables for 08 §1.2 (PCA and dimensionality reduction).

  python scripts/experiments/08_pca/build_head.py <head> <section heading> <intro line>
Hashes per head are listed below; columns in the agreed order, every cell train / test, the human
ceiling (mean and median of experts) as reference columns. Writes tmp/exp08_<head>.md.
"""
import re
import subprocess
import sys
from pathlib import Path

P = r"C:\Users\abi\miniconda3\envs\fyp-local\python.exe"
HASHES = {
    "lr": {"ctx": ["a2096ec10bef", "b3fbc8430d44", "5580383edd8d", "9990d1832233", "69cc580dcd9a", "617bf1d8b13b", "9cd26eabef2a"],
           "nctx": ["5d86ad5c10cf", "c724586518c0", "049b908f551c", "60fc84b50016", "b4476ab7dc49", "84227cefa9b3", "41511c48f91c"]},
    "rf": {"ctx": ["a1e44780a01f", "e36bbdfd5379", "4b2ab09d6733", "783d388f9d4b", "be9ae5a21b9f", "998051404e42", "c1e4357404df"],
           "nctx": ["c36f89504032", "ef5d328b5033", "a42cc72b352d", "bca4287f6f19", "9ee0dffa4325", "0967841280a6", "1a6494971aeb"]},
    "knn": {"ctx": ["9ecaa0ec1591", "2fcb719b3a54", "f4b118c8b4cd", "742c43adc8f7", "54378931db3d", "9545f014ad42", "7cfcbad29dcb"],
            "nctx": ["47550b2e37a7", "95a0f17474d2", "1c47d07d9025", "2fa68c93dbb6", "4551ccc115df", "659b59ec05eb", "02e4793a852a"]},
    "knn5": {"ctx": ["6015fa0595bc", "e79c4c117090", "4b839fb3be18", "32b20efcc0c1", "229aa6b96c09", "8720768ba51a", "88636418db8d"],
             "nctx": ["f47fa833d860", "bcbc82e02a33", "f6b45f12fea1", "370d203d04a5", "8f5cfb454394", "943933634b4a", "2cc43b4528c2"]},
}
NAMES = {"ctx": ["all 95 PCs", "all 95", "PCA 5", "PCA 10", "PCA 20", "PCA 90%", "PCA 95%"],
         "nctx": ["all 27 PCs", "all 27", "PCA 5", "PCA 10", "PCA 20", "PCA 90%", "PCA 95%"]}
SETTINGS = {"ctx": ["`expert.pca=95`", "–", "`expert.pca=5`", "`expert.pca=10`", "`expert.pca=20`", "`expert.pca=0.9`", "`expert.pca=0.95`"],
            "nctx": ["`expert.pca=27`", "–", "`expert.pca=5`", "`expert.pca=10`", "`expert.pca=20`", "`expert.pca=0.9`", "`expert.pca=0.95`"]}


def table(head: str, space: str, track: str) -> str:
    runs = [f"{n}={h}" for n, h in zip(NAMES[space], HASHES[head][space])]
    out = subprocess.run([P, "-m", "scripts.reporting.results_table", *runs, "--track", track, "--no-train-mean"],
                         capture_output=True, text=True, check=True).stdout
    out = re.sub(r" \(`[0-9a-f]{12}`\)", "", out)                 # hashes are in the arms table
    lines = [l for l in out.splitlines() if not l.startswith(r"\* threshold")]
    return "\n".join(lines).strip()


def main() -> None:
    head = sys.argv[1]
    arms = ["| Arm | With the class context (95 columns) | Without it (27 per-student columns) |", "|---|---|---|"]
    for i, (n, setting) in enumerate(zip(NAMES["ctx"], SETTINGS["ctx"])):
        n2 = NAMES["nctx"][i]
        arms.append(f"| {n} / {n2} | `{HASHES[head]['ctx'][i]}` | `{HASHES[head]['nctx'][i]}` |")
    parts = ["Results:", "", "\n".join(arms), "",
             "**With the class context (all 95 columns), expert entries:**", "", table(head, "ctx", "experts"), "",
             "**Without the class context (the 27 per-student columns), expert entries:**", "", table(head, "nctx", "experts"), "",
             "**With the class context, self-report entries:**", "", table(head, "ctx", "self"), "",
             "**Without the class context, self-report entries:**", "", table(head, "nctx", "self"), ""]
    Path(f"tmp/exp08_{head}.md").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
