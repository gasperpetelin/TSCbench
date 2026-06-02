"""Run a TSC benchmark with Rocket, MiniRocket, Catch22, and TSCGlue.

Examples:
    uv run python scripts/run_benchmark.py --datasets Crop --models tscglue --folds 0
    uv run python scripts/run_benchmark.py --datasets ArrowHead --models rocket,catch22
"""

# ruff: noqa: E402

import hashlib
import json
import os
import shutil
import sys
import tempfile
from itertools import product
from pathlib import Path
from time import perf_counter

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click
import numpy as np
import polars as pl
from aeon.classification.convolution_based import MiniRocketClassifier, RocketClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.datasets.tsc_datasets import univariate_equal_length
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tscglue.models import TSCGlueClassifier, TSCGlueLogisticClassifier

from tscbench.utils import (
    _DEFAULT_DATA_DIR,
    LocalFileCache,
    LogsFileCache,
    S3FileCache,
    discover_datasets,
    hardware_info,
    load_ucr_fold,
    software_versions,
)


class LogisticRegressionClassifier:
    def __init__(self, random_state=None, n_jobs=1, C=1.0, max_iter=1000):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        n = X.shape[0]
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X.reshape(n, -1))
        self._clf = LogisticRegression(
            random_state=self.random_state, n_jobs=self.n_jobs,
            C=self.C, max_iter=self.max_iter,
        )
        self._clf.fit(X_scaled, y)
        return self

    def predict(self, X):
        return self._clf.predict(self._scaler.transform(X.reshape(X.shape[0], -1)))

    def predict_proba(self, X):
        return self._clf.predict_proba(self._scaler.transform(X.reshape(X.shape[0], -1)))

    def get_params(self, deep=True):
        return {"random_state": self.random_state, "n_jobs": self.n_jobs, "C": self.C, "max_iter": self.max_iter}


class TSCGlueETClassifier(TSCGlueClassifier):
    """TSCGlue using an ExtraTrees stacker instead of RidgeCV / LogisticRegressionCV.

    ``TSCGlueClassifier.__init__`` doesn't pass ``stacking_models``, so the base
    falls back to ``[self.STACKING_MODEL]`` — overriding the class attribute is
    all that's needed to swap the meta-model.
    """

    STACKING_MODEL = "probability-et"


def run_metadata(n_jobs: int, n_gpus: int) -> dict:
    return {
        "n_jobs": n_jobs,
        "n_gpus": n_gpus,
        "hardware": hardware_info(),
        "versions": software_versions(),
    }


def get_model(
    model_name: str, random_state: int, n_jobs: int, n_gpus: int, runs_dir: str | None = None
):
    if model_name == "rocket":
        return RocketClassifier(random_state=random_state, n_jobs=n_jobs)
    if model_name == "minirocket":
        return MiniRocketClassifier(random_state=random_state, n_jobs=n_jobs)
    if model_name == "catch22":
        return Catch22Classifier(random_state=random_state, n_jobs=n_jobs)
    if model_name == "hivecote-1h":
        return HIVECOTEV2(
            random_state=random_state, n_jobs=n_jobs, time_limit_in_minutes=60
        )
    if model_name == "hivecote-4h":
        return HIVECOTEV2(
            random_state=random_state, n_jobs=n_jobs, time_limit_in_minutes=240
        )
    if model_name == "logreg":
        return LogisticRegressionClassifier(random_state=random_state, n_jobs=n_jobs)
    if model_name == "tscglue-logreg":
        return TSCGlueLogisticClassifier(
            random_state=random_state, n_jobs=n_jobs, n_gpus=n_gpus, runs_dir=runs_dir
        )
    if model_name == "tscglue-et":
        return TSCGlueETClassifier(
            random_state=random_state, n_jobs=n_jobs, n_gpus=n_gpus, runs_dir=runs_dir
        )
    if model_name == "tscglue":
        return TSCGlueClassifier(
            random_state=random_state, n_jobs=n_jobs, n_gpus=n_gpus, runs_dir=runs_dir
        )
    raise ValueError(f"Unknown model name: {model_name}")


def make_cache(storage: str, output_dir: Path, s3_uri: str):
    if storage == "s3":
        return S3FileCache(s3_uri)
    if storage == "logs":
        return LogsFileCache()
    return LocalFileCache(output_dir)


def _stable_digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def tscglue_runs_dir(
    dataset_name: str, model_name: str, fold: int, random_state: int, n_jobs: int, n_gpus: int
) -> Path:
    identity = {
        "dataset": dataset_name,
        "fold": fold,
        "model": model_name,
        "model_params": {
            "random_state": random_state,
            "n_jobs": n_jobs,
            "n_gpus": n_gpus,
        },
    }
    run_name = (
        f"{_safe_path_part(dataset_name)}_{_safe_path_part(model_name)}_"
        f"{fold}_{_stable_digest(identity)}"
    )
    return Path(tempfile.gettempdir()) / "tscglue-runs" / run_name


@click.command()
@click.option(
    "-m",
    "--models",
    multiple=True,
    required=True,
    help="Models to run. May be repeated or comma-separated.",
)
@click.option(
    "-d",
    "--datasets",
    "dataset_names",
    multiple=True,
    help="Datasets to run. May be repeated or comma-separated.",
)
@click.option(
    "-f",
    "--folds",
    "fold_spec",
    default="0",
    show_default=True,
    help="Folds to run, comma-separated, or 'all'.",
)
@click.option(
    "-o", "--output-dir", default="results", show_default=True, type=click.Path(path_type=Path)
)
@click.option(
    "--storage", type=click.Choice(["disk", "s3", "logs"]), default="logs", show_default=True
)
@click.option("--s3-uri", default="s3://tsc-bench/performance-benchmarking", show_default=True)
@click.option("-j", "--n-jobs", default=8, show_default=True, type=int)
@click.option("-g", "--n-gpus", default=0, show_default=True, type=int)
@click.option("--overwrite", is_flag=True)
@click.option("--list-datasets", is_flag=True)
def main(
    models,
    dataset_names,
    fold_spec,
    output_dir,
    storage,
    s3_uri,
    n_jobs,
    n_gpus,
    overwrite,
    list_datasets,
):
    local_datasets = discover_datasets(_DEFAULT_DATA_DIR)

    if list_datasets:
        for name in local_datasets:
            click.echo(f"{name} (30 folds)")
        return

    model_names = [m.strip() for ms in models for m in ms.split(",") if m.strip()]
    datasets = [
        d.strip() for ds in dataset_names for d in ds.split(",") if d.strip()
    ] or sorted(univariate_equal_length)

    cache = make_cache(storage, output_dir, s3_uri)
    metadata = run_metadata(n_jobs=n_jobs, n_gpus=n_gpus)

    click.echo(f"Models:   {', '.join(model_names)}")
    click.echo(f"Datasets: {', '.join(datasets)}")
    click.echo(f"Results:  {s3_uri if storage == 's3' else output_dir}")

    if fold_spec == "all":
        folds = list(range(30))
    else:
        folds = [int(f.strip()) for f in fold_spec.split(",") if f.strip()]

    triplets = list(product(model_names, datasets, folds))
    rng = np.random.default_rng()
    rng.shuffle(triplets)

    for model_name, dataset_name, fold in triplets:
            runs_dir = None
            if model_name in ("tscglue", "tscglue-logreg", "tscglue-et"):
                runs_dir = tscglue_runs_dir(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    fold=fold,
                    random_state=fold,
                    n_jobs=n_jobs,
                    n_gpus=n_gpus,
                )

            try:
                model = get_model(
                    model_name,
                    random_state=fold,
                    n_jobs=n_jobs,
                    n_gpus=n_gpus,
                    runs_dir=str(runs_dir) if runs_dir is not None else None,
                )
                model_params = {
                    k: str(v)
                    for k, v in model.get_params().items()
                    if not k.endswith("_dir")
                }

                stats = {
                    "dataset": dataset_name,
                    "fold": fold,
                    "model": model_name,
                    "random_state": fold,
                    **metadata,
                    "model_params": model_params,
                }

                cache_key = {
                    k: stats[k]
                    for k in (
                        "dataset",
                        "fold",
                        "model",
                        "random_state",
                        "n_jobs",
                        "hardware",
                        "versions",
                        "model_params",
                    )
                }
                cache_hash = (
                    pl.DataFrame([cache_key])
                    .hash_rows(seed=42, seed_1=1, seed_2=2, seed_3=3)
                    .item()
                )
                filename = f"{cache_hash}.parquet"

                if cache.exists(filename) and not overwrite:
                    click.echo(f"Skipping: dataset={dataset_name} fold={fold} model={model_name}")
                    continue

                if runs_dir is not None:
                    shutil.rmtree(runs_dir, ignore_errors=True)
                    runs_dir.mkdir(parents=True, exist_ok=True)

                click.echo(f"Running:  dataset={dataset_name} fold={fold} model={model_name}")

                X_train, y_train, X_test, y_test = load_ucr_fold(dataset_name, fold)
                stats["dataset_stats"] = {
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "n_classes": int(len(np.unique(y_train))),
                    "n_channels": int(X_train.shape[1]),
                    "n_timepoints": int(X_train.shape[2]),
                }

                t0 = perf_counter()
                model.fit(X_train, y_train)
                fit_s = perf_counter() - t0

                t0 = perf_counter()
                preds = model.predict(X_test)
                predict_s = perf_counter() - t0

                stats["timing"] = {"fit_s": fit_s, "predict_s": predict_s}
                stats["y_true"] = y_test.tolist()
                stats["y_pred"] = preds.tolist()
                if hasattr(model, "predict_proba"):
                    stats["y_prob"] = model.predict_proba(X_test).tolist()

                cache.add(pl.DataFrame([stats]), filename)
            except Exception as exc:
                import traceback
                click.echo(
                    f"Error: dataset={dataset_name} fold={fold} model={model_name}: {exc}\n"
                    + traceback.format_exc(),
                    err=True,
                )
            finally:
                if runs_dir is not None:
                    shutil.rmtree(runs_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
