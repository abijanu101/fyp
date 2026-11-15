"""Seeded random search over config-defined spaces.

Space format (JSON):  {"param": [choice, ...]}                         categorical
                      {"param": {"low": a, "high": b, "log": false, "int": false}}   range
Trial 0 always evaluates `baseline` (the current config), so every search reports its gain over it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..spu.splits import logo_splits


def sample_params(space: dict, rng: np.random.Generator) -> dict:
    out = {}
    for name, spec in sorted(space.items()):
        if isinstance(spec, list):
            out[name] = spec[int(rng.integers(len(spec)))]
            continue
        lo, hi = float(spec["low"]), float(spec["high"])
        v = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if spec.get("log") else float(rng.uniform(lo, hi))
        out[name] = int(round(v)) if spec.get("int") else round(v, 4)
    return out


def random_search(space: dict, objective: Callable[[dict], dict], n_trials: int, seed: int = 0,
                  baseline: dict | None = None, on_trial: Callable[[dict], None] | None = None) -> list[dict]:
    """objective(params) -> {"score": float, ...}. Returns trials sorted by score, best first."""
    rng = np.random.default_rng(seed)
    trials = []
    for i in range(n_trials):
        params = dict(baseline or {}) if i == 0 else sample_params(space, rng)
        result = {"trial": i, "params": params, **objective(params)}
        trials.append(result)
        if on_trial:
            on_trial(result)
    return sorted(trials, key=lambda r: -np.nan_to_num(r["score"], nan=-np.inf))


def _numeric_dims(space: dict, inactive: set[str]) -> list[str]:
    return sorted(k for k, s in space.items() if isinstance(s, dict) and k not in inactive)


def _to_unit(spec: dict, v: float) -> float:
    lo, hi = float(spec["low"]), float(spec["high"])
    if spec.get("log"):
        return float(np.clip((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1))
    return float(np.clip((v - lo) / (hi - lo), 0, 1))


def _from_unit(spec: dict, u: float):
    lo, hi = float(spec["low"]), float(spec["high"])
    v = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))) if spec.get("log") else lo + u * (hi - lo)
    return int(round(v)) if spec.get("int") else round(v, 4)


def beam_bisection_search(space: dict, objective: Callable[[dict], dict], n_trials: int, baseline: dict,
                          beam: int = 2, inactive: set[str] = frozenset(), min_width: float = 1 / 64,
                          on_trial: Callable[[dict], None] | None = None) -> list[dict]:
    """Deterministic branch-and-bound search: a binary search per numeric parameter, k branches per
    node, a beam as the bound.

    A node is a point plus, per numeric parameter, an interval around it (in [0, 1] units, log scale
    where the space says `log`). Expanding a node creates k + 1 children (k = numeric parameters):
      branch i     probe the midpoints of the lower and upper halves of parameter i's interval; keep
                   the best of the three points and halve the interval around it (bisection)
      combined     apply every branch's winning move at once (one extra trial)
    Only the `beam` best nodes of each level are expanded further (bound); the search stops when the
    trial budget is spent or every interval is narrower than `min_width`. Categorical parameters,
    and numeric ones listed in `inactive` (no effect under the fixed settings), keep their baseline
    values. Trial 0 is the baseline. Identical points are evaluated once. Returns trials best first,
    in the same format as random_search."""
    dims = _numeric_dims(space, set(inactive))
    trials, seen = [], {}

    def evaluate(params: dict) -> float:
        key = tuple(sorted((k, str(v)) for k, v in params.items()))
        if key in seen:
            return seen[key]
        if len(trials) >= n_trials:
            return -np.inf
        result = {"trial": len(trials), "params": dict(params), **objective(dict(params))}
        trials.append(result)
        if on_trial:
            on_trial(result)
        seen[key] = float(np.nan_to_num(result["score"], nan=-np.inf))
        return seen[key]

    def params_of(u: dict) -> dict:
        return {**baseline, **{d: _from_unit(space[d], u[d]) for d in dims}}

    start = {d: _to_unit(space[d], baseline[d]) if d in baseline else 0.5 for d in dims}
    root = {"u": start, "lo": {d: 0.0 for d in dims}, "hi": {d: 1.0 for d in dims},
            "score": evaluate(params_of(start))}
    frontier = [root]
    while frontier and len(trials) < n_trials:
        children = []
        for node in frontier:
            moves = {}
            for d in dims:
                lo, hi, u = node["lo"][d], node["hi"][d], node["u"][d]
                if hi - lo < min_width:
                    continue
                cands = [(node["score"], u, lo, hi)]
                for v, nlo, nhi in (((lo + u) / 2, lo, u), ((u + hi) / 2, u, hi)):
                    s = evaluate(params_of({**node["u"], d: v}))
                    cands.append((s, v, nlo, nhi))
                s, v, nlo, nhi = max(cands, key=lambda c: c[0])
                if v == u:                                   # staying put: halve the interval around u
                    w = (hi - lo) / 4
                    nlo, nhi = max(lo, u - w), min(hi, u + w)
                moves[d] = (v, nlo, nhi)
                children.append({"u": {**node["u"], d: v}, "lo": {**node["lo"], d: nlo},
                                 "hi": {**node["hi"], d: nhi}, "score": s})
            if moves and any(m[0] != node["u"][d] for d, m in moves.items()):
                u = {**node["u"], **{d: m[0] for d, m in moves.items()}}
                children.append({"u": u, "lo": {**node["lo"], **{d: m[1] for d, m in moves.items()}},
                                 "hi": {**node["hi"], **{d: m[2] for d, m in moves.items()}},
                                 "score": evaluate(params_of(u))})
        unique = {}
        for c in sorted(children, key=lambda c: -c["score"]):
            unique.setdefault(tuple(round(c["u"][d], 6) for d in dims), c)
        frontier = [c for c in list(unique.values())[:beam] if np.isfinite(c["score"])]
    return sorted(trials, key=lambda r: -np.nan_to_num(r["score"], nan=-np.inf))


def nested_logo_search(groups: np.ndarray, space: dict, n_trials: int, seed: int,
                       score: Callable[[dict, np.ndarray, np.ndarray], float],
                       baseline: dict | None = None) -> dict:
    """Hyperparameter search that does not select on its own test fold.

    score(params, train_idx, test_idx) -> float (higher is better). For each outer LOGO fold, the
    search runs inner LOGO on the outer-train groups only; the winner is refit and scored once on
    the outer test group. The `selected` params come from a final search over all groups and are
    the ones to put in the config; `outer` scores are the unbiased estimate of their quality.
    """
    def search(idx: np.ndarray) -> dict:
        def objective(params):
            s = [score(params, idx[tr], idx[te]) for _, tr, te in logo_splits(groups[idx])]
            return {"score": float(np.nanmean(s)), "fold_scores": s}
        return random_search(space, objective, n_trials, seed, baseline)[0]

    outer = []
    for g, tr, te in logo_splits(groups):
        best = search(tr)
        outer.append({"held_out": g, "params": best["params"], "inner_score": best["score"],
                      "outer_score": score(best["params"], tr, te)})
    final = search(np.arange(len(groups)))
    s = [o["outer_score"] for o in outer]
    return {"selected": final["params"], "selected_cv_score": final["score"],
            "nested_mean": float(np.nanmean(s)), "nested_std": float(np.nanstd(s)), "outer": outer}
