"""Classical classifiers shared by the SPU gate (§7) and the engagement experts (§8).

Every model is wrapped as impute(median) -> standardise -> [PCA] -> classifier, so NaN features
(e.g. windows with no visible tracks) are handled the same way everywhere. `pca` (optional) is
sklearn's n_components: an int keeps that many components, a float in (0, 1) keeps enough to explain
that share of the variance; it is fitted with the rest of the pipeline, on the training rows only.
"""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSIFIERS = {
    # fixed choices first, config params last: any of them can be overridden from the config
    "random_forest": lambda seed, **p: RandomForestClassifier(**{"random_state": seed, **p}),
    "svm": lambda seed, **p: SVC(**{"probability": True, "random_state": seed, **p}),
    "logreg": lambda seed, **p: LogisticRegression(**{"max_iter": 2000, "random_state": seed, **p}),
    "knn": lambda seed, **p: KNeighborsClassifier(**p),
}


def build_classifier(name: str, seed: int = 0, pca: int | float | None = None, **params) -> Pipeline:
    if name not in CLASSIFIERS:
        raise KeyError(f"Unknown classifier '{name}'. Options: {sorted(CLASSIFIERS)}")
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("scale", StandardScaler())]
    if pca:
        steps.append(("pca", PCA(n_components=pca, random_state=seed)))
    return Pipeline(steps + [("clf", CLASSIFIERS[name](seed, **params))])
