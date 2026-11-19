"""The 12 L1 base models of TSCGlue's `high` preset as standalone classifiers.

Each one is a single representation + a single head, fitted once on the full train
set: transform -> drop constant columns -> scale -> head. No cross-fitting, no
stacking, no fallback. Hyperparameters match tscglue's MODELS_DICTIONARY.

Nothing is imported from tscglue: the transforms only it provides (U-RSTSF
intervals, float64 WEASEL v2, Mantis and Chronos2 embeddings) are defined here.
"""

import warnings

import numpy as np
from aeon.classification.base import BaseClassifier
from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2
from aeon.transformations.collection import ARCoefficientTransformer, PeriodogramTransformer
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.transformations.collection.interval_based import QUANTTransformer, RandomIntervals
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from aeon.utils.numba.general import first_order_differences_3d
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2, f_classif
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.extmath import softmax
from threadpoolctl import threadpool_limits


class SoftmaxRidgeClassifierCV(RidgeClassifierCV):
    """RidgeClassifierCV with softmax-of-decision-function probabilities."""

    def predict_proba(self, X):
        scores = self.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        return softmax(scores)


class SparseScaler:
    """Hydra's scaler: sqrt of clipped counts, standardised over non-zero entries."""

    def __init__(self, exponent=4):
        self.exponent = exponent

    def fit_transform(self, X, y=None):
        Xt = np.sqrt(np.clip(X, 0, None))
        epsilon = (Xt == 0).mean(axis=0) ** self.exponent + 1e-8
        self.mu_ = Xt.mean(axis=0)
        self.sigma_ = Xt.std(axis=0) + epsilon
        return self.transform(X)

    def transform(self, X):
        Xt = np.sqrt(np.clip(X, 0, None))
        return (Xt - self.mu_) * (Xt != 0) / self.sigma_


def _no_scaler():
    return FunctionTransformer()


def _seed(rng):
    return int(rng.integers(0, 2**31 - 1))


class WEASELTransformerV2F64(WEASELTransformerV2):
    """Unsupervised WEASEL v2 (feature_selection="none") that casts input to float64.

    Upstream always copies y in fit_transform although feature_selection="none" never
    reads it, so a placeholder y stands in.
    """

    def fit_transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        return super().fit_transform(X, np.zeros(X.shape[0], dtype=int) if y is None else y)

    def _transform(self, X, y=None):
        return super()._transform(np.asarray(X, dtype=np.float64), y)


class _FastRandomIntervals:
    """aeon's RandomIntervals fit (same intervals_), with a vectorised transform."""

    def __init__(self, n_intervals, min_interval_length, random_state):
        self._ri = RandomIntervals(
            n_intervals=n_intervals, min_interval_length=min_interval_length,
            random_state=random_state, n_jobs=1,
        )

    def fit(self, X):
        self._ri.fit(X)
        return self

    def transform(self, X):
        intervals = self._ri.intervals_
        Xt = np.empty((X.shape[0], len(intervals)), dtype=X.dtype)
        for i, (start, end, dim, feature, dilation) in enumerate(intervals):
            Xt[:, i] = feature(X[:, dim, start:end:dilation])
        return Xt


class RSTSFRandomTransformer:
    """U-RSTSF: random (unsupervised) intervals over RSTSF's four series representations.

    The representations are the raw series, first differences, periodogram and AR
    coefficients, as in RSTSF, but intervals are drawn at random instead of selected
    with the labels.
    """

    def __init__(self, n_intervals=600, min_interval_length=3, random_state=None):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        lags = int(12 * (X.shape[2] / 100.0) ** 0.25)
        self.series_transformers_ = [
            FunctionTransformer(func=first_order_differences_3d, validate=False),
            PeriodogramTransformer(),
            ARCoefficientTransformer(order=lags, replace_nan=True),
        ]
        reps = [X] + [t.fit_transform(X) for t in self.series_transformers_]
        self.ri_ = [
            _FastRandomIntervals(
                self.n_intervals, self.min_interval_length, self.random_state
            ).fit(rep)
            for rep in reps
        ]
        return np.hstack([ri.transform(rep) for ri, rep in zip(self.ri_, reps)])

    def transform(self, X):
        reps = [X] + [t.transform(X) for t in self.series_transformers_]
        return np.hstack([ri.transform(rep) for ri, rep in zip(self.ri_, reps)])


class MantisEmbedding:
    """Frozen MantisV2 embeddings (256d) of the series and of its first differences,
    each resized to 512 time steps."""

    def __init__(self, device="cpu"):
        self.device = device

    def _embed(self, X):
        import torch
        import torch.nn.functional as F
        from mantis.architecture import MantisV2
        from mantis.trainer import MantisTrainer

        if not hasattr(self, "model_"):
            network = MantisV2(device=self.device).from_pretrained("paris-noah/MantisV2")
            self.model_ = MantisTrainer(device=self.device, network=network)
        X_r = F.interpolate(
            torch.tensor(X, dtype=torch.float), size=512, mode="linear", align_corners=False
        ).numpy()
        return self.model_.transform(X_r)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return np.hstack([self._embed(X), self._embed(np.diff(X, axis=-1))])


class Chronos2Embedding:
    """Frozen Chronos-2 [REG] token embeddings of the series and of its first differences."""

    def __init__(self, model_id="amazon/chronos-2", batch_size=256, device="cpu"):
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device

    def _embed(self, X):
        import torch
        from chronos import BaseChronosPipeline

        if not hasattr(self, "pipeline_"):
            self.pipeline_ = BaseChronosPipeline.from_pretrained(
                self.model_id, device_map=self.device
            )
        out = []
        with torch.inference_mode():
            for i in range(0, len(X), self.batch_size):
                batch = [torch.from_numpy(x.squeeze(0)).float() for x in X[i : i + self.batch_size]]
                embeddings, _ = self.pipeline_.embed(batch)
                # Chronos-2 is decoder-only: the [REG] token is second to last.
                out.append(np.stack([e[0, -2, :].cpu().float().numpy() for e in embeddings]))
        return np.vstack(out)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return np.hstack([self._embed(X), self._embed(np.diff(X, axis=-1))])


def _weasel(rng, n_jobs):
    return WEASELTransformerV2F64(
        min_window=4, norm_options=(False,), word_lengths=(7, 8),
        use_first_differences=(True, False), feature_selection="none",
        max_feature_count=30_000, random_state=_seed(rng), n_jobs=n_jobs,
    )


def _rstsf_random(rng, n_jobs):
    return RSTSFRandomTransformer(random_state=_seed(rng))


def _mantis(rng, n_jobs):
    return MantisEmbedding(device="cpu")


def _chronos2(rng, n_jobs):
    return Chronos2Embedding(device="cpu")


# L0 representations: transform name -> factory(rng, n_jobs).
REPRESENTATIONS = {
    "multirockethydra": {
        "multirocket": lambda rng, n_jobs: MultiRocket(n_jobs=n_jobs, random_state=_seed(rng)),
        "hydra": lambda rng, n_jobs: HydraTransformer(n_jobs=n_jobs, random_state=_seed(rng)),
    },
    "quant": {"quant": lambda rng, n_jobs: QUANTTransformer()},
    "rdst": {
        "rdst": lambda rng, n_jobs: RandomDilatedShapeletTransform(
            n_jobs=n_jobs, random_state=_seed(rng)
        ),
    },
    "rstsf-random": {"rstsf-random": _rstsf_random},
    "fm": {"mantis": _mantis, "chronos2": _chronos2},
    "weasel": {"weasel": _weasel},
}


def _ridge(alphas=np.logspace(-3, 3, 10)):
    return lambda seed, n_train, n_jobs: SoftmaxRidgeClassifierCV(alphas=alphas)


class LogisticCV(ClassifierMixin, BaseEstimator):
    """Multinomial logistic regression with C picked by 3-fold CV.

    Falls back to a plain LogisticRegression(C=1) when a class has fewer than 3 training
    cases, since stratified 3-fold CV cannot put it in every fold.
    """

    def __init__(self, Cs=np.logspace(-4, 1, 6), n_jobs=1):
        self.Cs = Cs
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if np.unique(y, return_counts=True)[1].min() < 3:
            self.estimator_ = LogisticRegression(C=1.0, max_iter=1000)
        else:
            self.estimator_ = LogisticRegressionCV(
                Cs=self.Cs, cv=3, max_iter=1000, n_jobs=self.n_jobs
            )
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


def _logreg():
    return lambda seed, n_train, n_jobs: LogisticCV(n_jobs=n_jobs)


def _bestk(clf, score_func, k=None):
    # k=None: tscglue's sigmoid in n_train, from 6000 up to 35000 features.
    def make(seed, n_train, n_jobs):
        kk = k if k is not None else int(
            6000 + (35000 - 6000) / (1.0 + np.exp(-0.010 * (n_train - 300)))
        )
        return Pipeline([
            ("select", SelectKBest(score_func=score_func, k=kk)),
            ("clf", clf(n_jobs)),
        ])
    return make


def _bestk_ridge(score_func, k=None):
    return _bestk(lambda n_jobs: SoftmaxRidgeClassifierCV(alphas=np.logspace(-3, 3, 10)), score_func, k)


def _bestk_logreg(score_func, k=None):
    return _bestk(lambda n_jobs: LogisticCV(n_jobs=n_jobs), score_func, k)


def _etc(**kwargs):
    params = {"n_estimators": 200, "criterion": "entropy", "max_features": "sqrt"} | kwargs
    return lambda seed, n_train, n_jobs: ExtraTreesClassifier(
        **params, random_state=seed, n_jobs=n_jobs
    )


def _rf(**kwargs):
    # Same settings as _etc, so RF vs ET differs only in bootstrapping and best-vs-random splits.
    params = {"n_estimators": 200, "criterion": "entropy", "max_features": "sqrt"} | kwargs
    return lambda seed, n_train, n_jobs: RandomForestClassifier(
        **params, random_state=seed, n_jobs=n_jobs
    )


# L1 base models: name -> (representation, scaler per transform, head factory).
BASE_MODELS = {
    "multirockethydra-bestk-p-ridgecv": (
        "multirockethydra", {"multirocket": StandardScaler, "hydra": SparseScaler},
        _bestk_ridge(f_classif),
    ),
    "multirockethydra-etc": (
        "multirockethydra", {"multirocket": _no_scaler, "hydra": _no_scaler}, _etc(),
    ),
    "quant-etc": ("quant", {"quant": _no_scaler}, _etc(max_features=0.1)),
    "quant-p-ridgecv": ("quant", {"quant": StandardScaler}, _ridge()),
    "rdst-p-ridgecv": ("rdst", {"rdst": StandardScaler}, _ridge(np.logspace(-4, 4, 20))),
    "rdst-etc": ("rdst", {"rdst": _no_scaler}, _etc()),
    "rstsf-random-etc": (
        "rstsf-random", {"rstsf-random": _no_scaler}, _etc(class_weight="balanced"),
    ),
    "rstsf-random-p-ridgecv": ("rstsf-random", {"rstsf-random": StandardScaler}, _ridge()),
    "fm-p-ridgecv": ("fm", {"mantis": StandardScaler, "chronos2": StandardScaler}, _ridge()),
    "fm-etc": ("fm", {"mantis": _no_scaler, "chronos2": _no_scaler}, _etc()),
    "weasel-bestk-p-ridgecv": ("weasel", {"weasel": _no_scaler}, _bestk_ridge(chi2, k=30000)),
    "weasel-etc": ("weasel", {"weasel": _no_scaler}, _etc()),
    # Not in TSCGlue: a Random Forest head per representation, with that representation's ET settings.
    "multirockethydra-rf": (
        "multirockethydra", {"multirocket": _no_scaler, "hydra": _no_scaler}, _rf(),
    ),
    "quant-rf": ("quant", {"quant": _no_scaler}, _rf(max_features=0.1)),
    "rdst-rf": ("rdst", {"rdst": _no_scaler}, _rf()),
    "rstsf-random-rf": ("rstsf-random", {"rstsf-random": _no_scaler}, _rf(class_weight="balanced")),
    "fm-rf": ("fm", {"mantis": _no_scaler, "chronos2": _no_scaler}, _rf()),
    "weasel-rf": ("weasel", {"weasel": _no_scaler}, _rf()),
    # Not in TSCGlue: logistic regression with exactly the RidgeCV head's scaling and feature
    # selection, so the two linear heads differ only in the loss and how probabilities are made.
    "multirockethydra-bestk-logreg": (
        "multirockethydra", {"multirocket": StandardScaler, "hydra": SparseScaler},
        _bestk_logreg(f_classif),
    ),
    "quant-logreg": ("quant", {"quant": StandardScaler}, _logreg()),
    "rdst-logreg": ("rdst", {"rdst": StandardScaler}, _logreg()),
    "rstsf-random-logreg": ("rstsf-random", {"rstsf-random": StandardScaler}, _logreg()),
    "fm-logreg": ("fm", {"mantis": StandardScaler, "chronos2": StandardScaler}, _logreg()),
    "weasel-bestk-logreg": ("weasel", {"weasel": _no_scaler}, _bestk_logreg(chi2, k=30000)),
}


class BaseModel(BaseClassifier):
    """One L1 base model from BASE_MODELS, fitted once on the full train set (CPU)."""

    _tags = {"capability:multivariate": False}

    def __init__(self, model_name, random_state=None, n_jobs=1):
        self.model_name = model_name
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__()

    def _features(self, X, fit):
        blocks = []
        for ft in self.transformers_:
            if fit:
                Xt = np.asarray(self.transformers_[ft].fit_transform(X), dtype=np.float64)
                Xt = self.scalers_[ft].fit_transform(self.filters_[ft].fit_transform(Xt))
            else:
                Xt = np.asarray(self.transformers_[ft].transform(X), dtype=np.float64)
                Xt = self.scalers_[ft].transform(self.filters_[ft].transform(Xt))
            blocks.append(Xt)
        return np.hstack(blocks)

    def _fit(self, X, y):
        representation, scalers, head = BASE_MODELS[self.model_name]
        rng = np.random.default_rng(self.random_state)
        self.transformers_ = {
            ft: make(rng, self.n_jobs) for ft, make in REPRESENTATIONS[representation].items()
        }
        self.filters_ = {ft: VarianceThreshold() for ft in self.transformers_}
        self.scalers_ = {ft: scalers[ft]() for ft in self.transformers_}
        # Cap BLAS threads at n_jobs, as tscglue does, so parallel jobs don't oversubscribe.
        with threadpool_limits(limits=self.n_jobs), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xt = self._features(X, fit=True)
            self.clf_ = head(_seed(rng), X.shape[0], self.n_jobs)
            if isinstance(self.clf_, Pipeline):
                self.clf_.set_params(select__k=min(self.clf_["select"].k, Xt.shape[1]))
            self.clf_.fit(Xt, y)
        return self

    def _predict_proba(self, X):
        with threadpool_limits(limits=self.n_jobs), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.clf_.predict_proba(self._features(X, fit=False))

    def _predict(self, X):
        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]
