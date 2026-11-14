# Chapter 2 — The Python toolkit

The codebase uses a small set of Python features over and over. None are exotic, but if one is
unfamiliar, the code that depends on it will look like magic. This chapter covers each with a
tiny standalone example, then shows where it appears in the codebase.

## 2.1 Packages, modules and relative imports

A **module** is one `.py` file. A **package** is a folder containing an `__init__.py`. Our code
lives in one top-level package, `src`, with one sub-package per pipeline stage:

```
src/
├── __init__.py            makes src a package
├── core/
│   ├── __init__.py
│   ├── types.py           module src.core.types
│   └── registry.py        module src.core.registry
└── bfmu/
    ├── __init__.py
    └── kalman_geometry.py module src.bfmu.kalman_geometry
```

Inside the package, modules import each other with **relative imports**:

```python
# in src/bfmu/kalman_geometry.py
from ..core.types import Tracklet          # ".." = go up one package (to src), then into core
from .base import FEATURE_EXTRACTORS       # "."  = this package (src.bfmu)
```

Relative imports only work when Python knows the file is part of a package. That is why scripts
are run with `-m`:

```
python -m scripts.run_pipeline extract     # works: run as module scripts.run_pipeline
python scripts/run_pipeline.py extract     # fails: "attempted relative import" or "No module named src"
```

`python -m` puts the current directory (the repo root) on the import path, so `import src...`
works from any script. The standalone tools in `scripts/dipser/` that do not import `src`
(`label_viewer.py`, `plot_labels.py`, `player.py`, `annotation_helper.py`, `camera_previews.py`)
can be run directly as files.

### What `__init__.py` files do here

Most of our `__init__.py` files do two things: re-export the names other packages need, and
import the modules whose import has a side effect. For example `src/bfmu/__init__.py`:

```python
from .base import FEATURE_EXTRACTORS, FeatureExtractor
from . import kalman_geometry, residual_flow, scb  # noqa: F401  (registers extractors)
```

The second line looks unused (hence `# noqa: F401`, which silences the linter's "unused import"
warning), but it is essential. Chapter 3 explains why.

## 2.2 `from __future__ import annotations`

Almost every module starts with this line. It makes type hints lazy: they are stored as strings
instead of being evaluated when the function is defined. Two practical effects:

- You can write modern hints like `list[int] | None` everywhere.
- A class can mention itself in its own hints (`def slice(...) -> "Tracklet"`) without errors.

It changes nothing at runtime. Type hints in this codebase are documentation; nothing enforces
them.

## 2.3 Dataclasses

A dataclass is a class whose job is to hold data. The decorator writes `__init__`, `__repr__` and
`__eq__` for you from the annotated fields:

```python
from dataclasses import dataclass

@dataclass
class Window:
    t_start: float
    t_end: float

w = Window(10.0, 18.0)
w.t_end            # 18.0
print(w)           # Window(t_start=10.0, t_end=18.0)
```

Two options matter in our code:

- `@dataclass(frozen=True)` makes instances read-only and hashable, so they can be dictionary
  keys or set members. `SessionKey` and `Window` are frozen.
- `field(default_factory=dict)` gives each instance its own fresh dict. Writing `= {}` as a
  default would share one dict between all instances, which is a classic bug.

`src/core/types.py` is almost entirely dataclasses. They are the **contracts** between stages:
Chapter 4 covers every one.

## 2.4 Abstract base classes

An abstract base class (ABC) defines what methods a family of classes must have, without
implementing them all:

```python
from abc import ABC, abstractmethod

class Tracker(ABC):
    @abstractmethod
    def update(self, frame, dets): ...

class Broken(Tracker):
    pass

Broken()     # TypeError: Can't instantiate abstract class Broken with abstract method update
```

Python refuses to create an instance of a class that has not implemented every abstract method.
The error appears when you construct the object, not later when the missing method would be
called. That is useful: a half-written stage fails loudly at pipeline construction.

Each stage has one ABC: `PersonDetector`, `Tracker`, `FeatureExtractor`, `SceneFeatureExtractor`,
`ClusterFeatureExtractor`, `EngagementExpert`, `FrameSource`. An ABC can also provide default
methods. `FeatureExtractor.frame_signals` returns `{}` by default, so extractors that do not need
pixels simply do not override it.

## 2.5 Decorators

A decorator is a function that takes a function or class and returns something to use in its
place. The `@` syntax is shorthand:

```python
@my_decorator
class Thing: ...

# means exactly:
class Thing: ...
Thing = my_decorator(Thing)
```

A decorator can also take arguments. Then there are two calls: the outer call receives the
arguments and returns the actual decorator:

```python
@register("botsort")
class BotSort: ...

# means:
deco = register("botsort")     # returns a function
BotSort = deco(BotSort)        # which is applied to the class
```

This two-step shape is exactly how our registries work (Chapter 3). Python's own `@dataclass`,
`@property`, `@abstractmethod`, `@staticmethod` and `@contextmanager` are also decorators.

## 2.6 `functools.partial`

`partial` pre-fills some arguments of a function or class, producing a new callable:

```python
from functools import partial

def power(base, exp):
    return base ** exp

square = partial(power, exp=2)
square(5)          # 25
```

In `src/spu/gate.py`, one class `ContextGate` serves four registry names by pre-filling the model
name:

```python
for _name in CLASSIFIERS:                     # "random_forest", "svm", "logreg", "knn"
    GATES.register(_name)(partial(ContextGate, _name))
```

`GATES.build("svm", classes=..., seed=0)` then calls `ContextGate("svm", classes=..., seed=0)`.

## 2.7 Context managers

A `with` block runs setup code before and cleanup code after, even if an error happens inside:

```python
with zipfile.ZipFile(path) as zf:     # opened here
    data = zf.read(name)
# closed here, no matter what
```

We write one ourselves: `StageTimer.stage` in `src/evaluation/timer.py` measures how long a block
takes, using `@contextmanager`:

```python
@contextmanager
def stage(self, name: str):
    start = time.perf_counter()
    try:
        yield                                   # the body of the with-block runs here
    finally:
        self.totals[name] += time.perf_counter() - start
```

Used as:

```python
with timer.stage("detection"):
    dets = self.detector.detect(frame)
```

## 2.8 Generators

A function with `yield` produces values one at a time instead of building a list. The frame
source is a generator: it decodes one frame, hands it to the pipeline, and only decodes the next
when asked. A 5-minute session at 1280×720 is about 4 GB of decoded pixels; a generator keeps one
frame in memory at a time.

```python
def __iter__(self):
    with zipfile.ZipFile(self.path) as zf:
        for i in range(len(self._members)):
            frame = self._decode(zf, i)
            if frame is not None:
                yield frame
```

## 2.9 numpy idioms that appear everywhere

A few patterns recur so often that they are worth recognising on sight:

```python
a[m]                        # boolean mask: keep rows where m is True
np.flatnonzero(m)           # indices where m is True
np.searchsorted(t, x)       # position where x would be inserted in sorted t (time lookups)
a[:, None] - b[None, :]     # broadcasting: all pairs, shape (len(a), len(b))
np.nanmean(x)               # mean ignoring NaN
```

**NaN is used on purpose** in this codebase. It means "no value here": a window with no visible
students has NaN for every per-student average. The models' preprocessing replaces NaN with the
training median (Chapter 7). Do not replace NaN with 0 early: 0 is a real value ("perfectly
still"), NaN is "unknown".

Broadcasting powers the box geometry. `iou_matrix` in `src/tracking/association.py` computes the
overlap of every box in `a` with every box in `b` in one expression, by indexing with `[:, None]`
and `[None, :]`.

## 2.10 What you can skip

The codebase avoids metaclasses, multiple inheritance, async code, and type-checking at runtime.
If you see something that looks clever, it is almost certainly one of the features above.
