# Chapter 3 — The registry pattern, from scratch

This chapter builds the registry pattern up one step at a time, starting from the naive code you
would write without it. By the end you will be able to trace exactly how the text `"botsort"` in
the config file becomes a running BoT-SORT tracker, and you will know how to add your own
component.

## 3.1 The problem

The design asks for many interchangeable parts:

- two trackers (StrongSORT and BoT-SORT), compared head-to-head;
- several feature extractors, which can be switched on and off;
- four gate models (Random Forest, SVM, logistic regression, k-NN);
- three expert types (logistic regression, Random Forest, hand-weighted baseline).

And the design wants every experiment to be reproducible from a config file: "run with BoT-SORT
and an SVM gate" should be a change of *data*, not of *code*.

So we need a way to go from a **name in a JSON file** to a **constructed object**, for every kind
of component.

## 3.2 Attempt 1: the if/elif factory

The first thing anyone writes:

```python
def make_tracker(name, **options):
    if name == "botsort":
        from boxmot import create_tracker
        return BotSortWrapper(create_tracker("botsort", **options))
    elif name == "strongsort":
        from boxmot import create_tracker
        return StrongSortWrapper(create_tracker("strongsort", **options))
    else:
        raise ValueError(f"unknown tracker {name}")
```

It works, but look at what happens when you add a third tracker:

1. You write the new tracker class in its own file.
2. You **also** edit `make_tracker` to add an `elif`.
3. You do the same for every other component type: one factory function each, each with its own
   chain of `elif`s, its own error message, its own way of passing options.

The factory has to know every implementation. Every new component means editing a central file
that someone else may also be editing. This is the coupling we want to remove.

## 3.3 Attempt 2: a dictionary

Replace the `if` chain with a lookup table:

```python
TRACKERS = {
    "botsort": BotSort,
    "strongsort": StrongSort,
}

def make_tracker(name, **options):
    if name not in TRACKERS:
        raise ValueError(f"unknown tracker {name}; known: {sorted(TRACKERS)}")
    return TRACKERS[name](**options)
```

Better: the factory no longer has any per-tracker logic. Construction is just "look up the class,
call it with the options". The error message lists what *is* available, which saves a lot of
debugging.

But the dictionary still has to list every class, so adding a tracker still means editing the
file where the dictionary lives. We have moved the coupling, not removed it.

## 3.4 Attempt 3: let each class add itself

What if each class put itself into the dictionary, right where it is defined?

```python
# tracking/tracker.py
TRACKERS = {}

class BotSort(Tracker):
    ...
TRACKERS["botsort"] = BotSort

class StrongSort(Tracker):
    ...
TRACKERS["strongsort"] = StrongSort
```

Now the knowledge "this class is called `botsort`" lives next to the class. A new tracker in a
new file adds itself; nothing central changes.

The `TRACKERS["botsort"] = BotSort` line after each class is easy to forget, and it is far from
the `class` line in a long file. A decorator puts it on top of the class where you cannot miss it.

## 3.5 Attempt 4: a decorator

Recall from Chapter 2 that `@deco(arg)` above a class means `Cls = deco(arg)(Cls)`. So we need a
function that takes a name and returns a function that stores the class and gives it back
unchanged:

```python
TRACKERS = {}

def register_tracker(name):
    def deco(cls):
        TRACKERS[name] = cls     # side effect: remember the class under this name
        return cls               # give the class back unchanged, so BotSort still means the class
    return deco

@register_tracker("botsort")
class BotSort(Tracker):
    ...
```

Read it as plain English: "register this class as the tracker called `botsort`". The class
itself is untouched; `BotSort` is still an ordinary class you can import and use directly (the
tests do exactly that with registered classes such as `NullClusterExtractor` and
`HandWeightedExpert`).

## 3.6 Attempt 5: one Registry class for every kind of component

We have many component types. Rather than writing `register_tracker`, `register_detector`,
`register_gate` and so on, we write the pattern once as a class and make one instance per type.
This is the actual code in `src/core/registry.py`:

```python
class Registry:
    def __init__(self, kind: str):
        self.kind = kind                                   # e.g. "tracker", used in error messages
        self._items: dict[str, Callable[..., Any]] = {}    # name -> class (or any callable)

    def register(self, name: str):
        def deco(obj):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._items[name] = obj
            return obj
        return deco

    def build(self, name: str, **kwargs) -> Any:
        return self.get(name)(**kwargs)

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Registered: {sorted(self._items)}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)
```

Four methods, about twenty lines:

| Method | What it does |
|---|---|
| `register(name)` | Returns a decorator that stores a class under `name`. Refuses duplicates. |
| `get(name)` | Returns the stored class (not an instance). Lists known names on a typo. |
| `build(name, **kwargs)` | `get(name)(**kwargs)`: looks up the class and constructs it. |
| `names()` | Lists what is registered, for error messages and debugging. |

Each stage package creates its own registry:

```python
# src/tracking/tracker.py
TRACKERS = Registry("tracker")

@TRACKERS.register("botsort")
class BotSort(BoxmotTracker):
    name = "botsort"

@TRACKERS.register("strongsort")
class StrongSort(BoxmotTracker):
    name = "strongsort"
```

And anyone can now construct a tracker by name:

```python
tracker = TRACKERS.build("botsort", use_embeddings=False, max_lost_frames=17)
```

Why is the duplicate check there? Two classes registered under the same name is always a bug,
usually a copy-pasted file where someone forgot to rename the decorator argument. Without the
check, the second silently replaces the first, and the pipeline runs the wrong code with no error.

## 3.7 Every registry in the codebase

| Registry | Defined in | Registered names | Contract (ABC) |
|---|---|---|---|
| `FRAME_SOURCES` | `src/ingestion/source.py` | `zip` | `FrameSource` |
| `DETECTORS` | `src/detection/detector.py` | `yolo` | `PersonDetector` |
| `TRACKERS` | `src/tracking/tracker.py` | `botsort`, `strongsort` | `Tracker` |
| `FEATURE_EXTRACTORS` | `src/bfmu/base.py` | `kalman_geometry`, `residual_flow`, `scb` | `FeatureExtractor` |
| `CLUSTER_EXTRACTORS` | `src/cfmu/extractor.py` | `null` | `ClusterFeatureExtractor` |
| `SCENE_EXTRACTORS` | `src/spu/scene.py` | `global_stats` | `SceneFeatureExtractor` |
| `GATES` | `src/spu/gate.py` | `random_forest`, `svm`, `logreg`, `knn` | (ContextGate) |
| `EXPERTS` | `src/experts/expert.py` | `logreg`, `random_forest`, `hand_weighted` | `EngagementExpert` |

## 3.8 The trap: registration happens at import time

The decorator runs when Python *executes the `class` statement*, which happens when the module is
*imported*. If nothing imports `src/bfmu/scb.py`, the `@FEATURE_EXTRACTORS.register("scb")` line
never runs, and:

```python
FEATURE_EXTRACTORS.build("scb", ...)
# KeyError: Unknown BFMU feature extractor 'scb'. Registered: ['kalman_geometry', 'residual_flow']
```

This is why each package's `__init__.py` imports its implementation modules, even though nothing
uses those names directly:

```python
# src/bfmu/__init__.py
from .base import FEATURE_EXTRACTORS, FeatureExtractor
from . import kalman_geometry, residual_flow, scb  # noqa: F401  (registers extractors)
```

When anything does `from ..bfmu import FEATURE_EXTRACTORS`, Python runs this `__init__.py`,
which imports the three modules, which register their classes. By the time you hold
`FEATURE_EXTRACTORS`, it is full.

**Rule:** a new implementation module must be imported in its package's `__init__.py`. If you
create `src/bfmu/pose.py` and forget this line, your extractor does not exist as far as the
pipeline is concerned. The error message, which lists what *is* registered, is your hint.

## 3.9 From JSON to object: the `_build` helper

The config file describes components as JSON objects with a `"name"` key plus constructor
arguments:

```json
"bfmu": [
  {"name": "kalman_geometry", "still_threshold": 0.05, "use_kalman": true},
  {"name": "residual_flow", "grid": 2, "scale": 0.5}
]
```

`src/core/pipeline.py` has a three-line helper that turns one such entry into an object:

```python
def _build(registry, entry: dict):
    """Build from a {"name": ..., **kwargs} config entry."""
    kwargs = dict(entry)                              # copy, so the config is not modified
    return registry.build(kwargs.pop("name"), **kwargs)
```

For the first entry above this does:

```python
kwargs = {"name": "kalman_geometry", "still_threshold": 0.05, "use_kalman": True}
name = kwargs.pop("name")                    # "kalman_geometry"; kwargs now has the other two
FEATURE_EXTRACTORS.build("kalman_geometry", still_threshold=0.05, use_kalman=True)
# == KalmanGeometry(still_threshold=0.05, use_kalman=True)
```

So **the keys of a config entry are exactly the constructor's keyword arguments**. To find out
what you can put in a config entry, read that class's `__init__`. A misspelled key produces a
clear `TypeError: __init__() got an unexpected keyword argument 'stil_threshold'`.

Not every stage uses `_build`. The tracker's config keeps BoxMOT's options in a nested
`"options"` object, because those are BoxMOT's parameters rather than ours:

```python
def new_tracker(self):
    return TRACKERS.build(self.cfg["tracker"]["name"], **self.cfg["tracker"].get("options", {}))
```

The gate and experts are built by the scripts, again with a separate `"params"` object for the
scikit-learn parameters.

## 3.10 Walkthrough: what `Pipeline(cfg)` actually does

Here is `Pipeline.__init__`, from `src/core/pipeline.py`:

```python
class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hash = config_hash(cfg)
        self._detector = None
        self.extractors = [_build(FEATURE_EXTRACTORS, e) for e in cfg["bfmu"]]
        self.scene = _build(SCENE_EXTRACTORS, cfg["scene"])
        agg = cfg["aggregation"]
        self.aggregator = WindowAggregator(
            self.extractors, _build(CLUSTER_EXTRACTORS, cfg["cfmu"]),
            thresholds=agg.get("thresholds"), heatmap_grid=agg.get("heatmap_grid", 3),
            min_coverage=agg.get("min_coverage", 0.5))
```

Step by step, with the current config:

1. **Imports already happened.** The `from ..bfmu import FEATURE_EXTRACTORS` at the top of
   `pipeline.py` ran `src/bfmu/__init__.py`, which registered all three extractors.
2. **`self.hash`**: a fingerprint of the whole config (Chapter 4).
3. **`self._detector = None`**: the detector is *not* built yet. Building YOLO loads weights onto
   the GPU, which the tests and the feature pass do not need. The `detector` property builds it
   on first use (section 3.12).
4. **`self.extractors`**: for each of the three `bfmu` entries, `_build` looks up the class and
   constructs it: `KalmanGeometry(still_threshold=0.05, use_kalman=True)`,
   `ResidualFlow(grid=2, scale=0.5)`, `ScbBehaviour(mode="soft", min_overlap=0.5, models=[...])`.
   The list keeps config order; the order matters only for the key order in Vec and the CSV. The
   models select columns by name (sorted), so it never affects training.
5. **`self.scene`**: `GlobalStats(size=[64, 36], grid=3)`.
6. **`self.aggregator`**: built directly (there is only one kind), receiving the extractors and a
   CFMU extractor built from `{"name": "null"}`, which is `NullClusterExtractor()`.

The tracker is built later, once per session, by `new_tracker()`, because a tracker carries state
(its current tracks) that must start fresh for each recording.

## 3.11 Contracts: why the ABC matters as much as the registry

The registry answers "how do I get an object by name". The abstract base class answers "what can
I do with it". Together they let the pipeline treat every extractor the same way, whichever one
the config chose:

```python
# src/bfmu/base.py
class FeatureExtractor(ABC):
    keys: tuple[str, ...] = ()        # the feature names this extractor produces
    uses_frames: bool = False         # does frame_signals read pixels?

    def reset(self) -> None:
        """Clear per-session state before a new session starts."""

    def frame_signals(self, frame, tracks) -> dict[int, dict[str, float]]:
        return {}

    @abstractmethod
    def window_features(self, tr: Tracklet, fps: float) -> dict[str, float]: ...
```

The pipeline calls only these members: `keys`, `uses_frames`, `reset()`, `frame_signals()`,
`window_features()`. It never checks *which* extractor it holds. So a new extractor plugs in with
no change anywhere else, as long as it honours the contract.

This is the whole design rule from §12.1 of the design document:

> Each stage is an abstract base class plus a name registry. A module is swapped by registering a
> new class and naming it in `configs/*.json`. Stages communicate only through the dataclasses in
> `core/types.py`, so a replacement has to keep its contract and nothing else.

## 3.12 Lazy construction

Some components are expensive to construct: YOLO loads weights onto the GPU; the SCB extractor
loads a 37-million-parameter YOLOv7. If constructing the pipeline loaded them, every unit test
would need a GPU and the weights files, and the feature pass (which never runs a network) would
waste seconds loading models it never uses.

So these use a **lazy property**: the attribute is built the first time it is read.

```python
@property
def detector(self):
    """Built on first use so the pipeline can be constructed (and tested) without a GPU."""
    if self._detector is None:
        self._detector = _build(DETECTORS, self.cfg["detector"])
    return self._detector
```

`ScbBehaviour.detectors` does the same. A consequence worth knowing: configuration mistakes in
these components (a wrong weights path, a class-name mismatch) surface when the first frame is
processed, not when the pipeline is built.

## 3.13 Families: one class, many names

Sometimes several registry names differ only in one argument. The four gate models are all "a
scikit-learn pipeline around a classifier"; only the classifier differs. Rather than four
near-identical classes, `src/spu/gate.py` registers one class four times with `partial`:

```python
class ContextGate:
    def __init__(self, model: str, classes: list[str], seed: int = 0, **params):
        self.classes = list(classes)
        self.model = build_classifier(model, seed, **params)
    ...

for _name in CLASSIFIERS:                          # random_forest, svm, logreg, knn
    GATES.register(_name)(partial(ContextGate, _name))
```

Notice `GATES.register(_name)(...)` called as a plain function, not as `@` syntax. That is the
same thing (section 2.5), written this way because it runs in a loop.

The registry stores the `partial` object, which behaves like a class: calling it constructs a
`ContextGate` with `model` already filled in. `src/experts/expert.py` does the same for the
`logreg` and `random_forest` experts.

## 3.14 How to add a component: worked example

Suppose you want a new BFMU feature family: **bounding-box jitter**, the frame-to-frame standard
deviation of each track's box width, as a fidgeting proxy. Here is the complete process.

**Step 1: write the class** in `src/bfmu/jitter.py`:

```python
from __future__ import annotations

import numpy as np

from ..core.types import Tracklet
from .base import FEATURE_EXTRACTORS, FeatureExtractor


@FEATURE_EXTRACTORS.register("box_jitter")
class BoxJitter(FeatureExtractor):
    keys = ("jit.width",)

    def __init__(self, normalise: bool = True):
        self.normalise = normalise

    def window_features(self, tr: Tracklet, fps: float) -> dict[str, float]:
        w = tr.xyxy[:, 2] - tr.xyxy[:, 0]
        if len(w) < 3:
            return {"jit.width": np.nan}
        jitter = np.std(np.diff(w))
        return {"jit.width": float(jitter / w.mean() if self.normalise else jitter)}
```

It needs no `frame_signals` (it only reads boxes, which the tracklet already has), so
`uses_frames` stays `False`.

**Step 2: make sure it gets imported**, in `src/bfmu/__init__.py`:

```python
from . import kalman_geometry, residual_flow, scb, jitter  # noqa: F401  (registers extractors)
```

**Step 3: switch it on** in `configs/deliverable3.json`:

```json
"bfmu": [
  {"name": "kalman_geometry", "still_threshold": 0.05, "use_kalman": true},
  {"name": "residual_flow", "grid": 2, "scale": 0.5},
  {"name": "scb", ...},
  {"name": "box_jitter", "normalise": true}
]
```

**Step 4: test it** (Chapter 10 shows the pattern): build a synthetic `Tracklet` with known
widths and check the number.

That is all. You did not touch the pipeline, the aggregator, the models or the scripts:

- the aggregator reads `extractor.keys` and adds `jit.width.mean` and `jit.width.std` to Vec;
- the experts are retrained on the wider Vec and pick up the new columns by name;
- because `BoxJitter` does not use frames, the feature hash changes but the tracking hash does
  not. `run_pipeline extract --reuse` recomputes features from cached tracks in seconds, without
  re-running detection (Chapter 8).

## 3.15 Why not something fancier?

There are heavier ways to do this: plugin systems with package entry points, automatic discovery
by scanning folders, dependency-injection frameworks. They solve problems this project does not
have (third parties shipping plugins, hundreds of components). The explicit version has
advantages that matter more here:

- **Everything is greppable.** Search for `register("scb")` and you find the class.
- **Nothing happens implicitly** except the one import rule in section 3.8.
- **It is twenty lines**, all visible in one file, and unit-testable.

## 3.16 Summary

- A **registry** maps names to classes. Classes add themselves with `@REGISTRY.register("name")`.
- `REGISTRY.build("name", **kwargs)` constructs one; config entries supply the kwargs.
- **Registration happens at import**, so each package's `__init__.py` imports its modules.
- The **ABC** is the contract: the pipeline only uses the ABC's methods, so any registered class
  that honours it plugs in.
- **Expensive components are built lazily**, on first use.
- **Adding a component** is: write class, import it in `__init__.py`, name it in config, test it.
