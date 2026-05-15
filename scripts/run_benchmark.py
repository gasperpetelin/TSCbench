"""Run a small TSC benchmark with Rocket, TSCGlue, and Catch22.

Examples:
    uv run python scripts/run_benchmark.py --datasets Crop --models tscglue --folds 0
    uv run python scripts/run_benchmark.py --datasets ArrowHead --models rocket,catch22
"""

# ruff: noqa: E402

import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click
import numpy as np
import polars as pl
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.datasets import load_from_ts_file
from sklearn.metrics import accuracy_score
from tscglue.data_loader import DATA_DIR as TSCGLUE_DATA_DIR
from tscglue.data_loader import load_fold as load_tscglue_fold
from tscglue.models import TSCGlue
from tscbench.utils import LocalFileCache, S3FileCache


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def run_metadata(n_jobs: int) -> dict:
    return {
        "python_version": sys.version.split()[0],
        "n_jobs": n_jobs,
        "tscbench_version": package_version("tscbench"),
        "tscglue_version": package_version("tscglue"),
        "aeon_version": package_version("aeon"),
        "sklearn_version": package_version("scikit-learn"),
        "numpy_version": package_version("numpy"),
        "polars_version": package_version("polars"),
    }


def get_model(model_name: str, random_state: int, n_jobs: int):
    if model_name == "rocket":
        return RocketClassifier(random_state=random_state, n_jobs=n_jobs)
    if model_name == "tscglue":
        return TSCGlue(random_state=random_state, n_jobs=n_jobs)
    if model_name == "catch22":
        return Catch22Classifier(random_state=random_state)
    raise ValueError(f"Unknown model name: {model_name}")


def expand_csv(values: tuple[str, ...]) -> list[str]:
    items = []
    for value in values:
        items.extend(item.strip() for item in value.split(",") if item.strip())
    return items


def discover_datasets(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        return []
    return sorted(path.name for path in data_dir.iterdir() if path.is_dir())


def discover_folds(data_dir: Path, dataset_name: str) -> list[int]:
    dataset_dir = data_dir / dataset_name
    if not dataset_dir.exists():
        return []

    pattern = re.compile(rf"^{re.escape(dataset_name)}(\d+)_TRAIN\.ts$")
    folds = []
    for path in dataset_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            folds.append(int(match.group(1)))
    return sorted(folds)


def load_local_fold(data_dir: Path, dataset_name: str, fold: int):
    dataset_dir = data_dir / dataset_name
    train_path = dataset_dir / f"{dataset_name}{fold}_TRAIN.ts"
    test_path = dataset_dir / f"{dataset_name}{fold}_TEST.ts"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing local fold files: {train_path.name} and/or {test_path.name}"
        )

    X_train, y_train = load_from_ts_file(train_path)
    X_test, y_test = load_from_ts_file(test_path)
    return X_train, y_train, X_test, y_test


def load_benchmark_fold(data_dir: Path, dataset_name: str, fold: int):
    if dataset_name.startswith("m-"):
        return load_tscglue_fold(dataset_name, fold)

    local_train = data_dir / dataset_name / f"{dataset_name}{fold}_TRAIN.ts"
    local_test = data_dir / dataset_name / f"{dataset_name}{fold}_TEST.ts"
    if local_train.exists() and local_test.exists():
        return load_local_fold(data_dir, dataset_name, fold)

    return load_tscglue_fold(dataset_name, fold)


def result_filename(dataset: str, model_name: str, fold: int) -> str:
    stem = f"{dataset}__{model_name}__fold_{fold}"
    safe_stem = re.sub(r"[^A-Za-z0-9_.+-]+", "_", stem)
    return f"{safe_stem}.parquet"


def make_cache(storage: str, output_dir: Path, s3_uri: str):
    if storage == "s3":
        return S3FileCache(s3_uri)
    return LocalFileCache(output_dir)


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
    "--data-dir",
    default="data",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Local fold data root. Falls back to tscglue's bundled data loader.",
)
@click.option(
    "-o",
    "--output-dir",
    default="artifacts/results",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory for per-run parquet result files.",
)
@click.option(
    "--storage",
    type=click.Choice(["disk", "s3"]),
    default="disk",
    show_default=True,
    help="Result cache backend.",
)
@click.option(
    "--s3-uri",
    default="s3://tsc-bench/performance-benchmarking",
    show_default=True,
    help="S3 result cache URI when --storage=s3.",
)
@click.option("-j", "--n-jobs", default=8, show_default=True, type=int)
@click.option("--overwrite", is_flag=True, help="Re-run even if a result parquet already exists.")
@click.option("--list-datasets", is_flag=True, help="List local datasets and exit.")
def main(
    models: tuple[str, ...],
    dataset_names: tuple[str, ...],
    fold_spec: str,
    data_dir: Path,
    output_dir: Path,
    storage: str,
    s3_uri: str,
    n_jobs: int,
    overwrite: bool,
    list_datasets: bool,
):
    local_data_dir = data_dir if data_dir.exists() else Path(TSCGLUE_DATA_DIR)
    local_datasets = discover_datasets(local_data_dir)

    if list_datasets:
        if not local_datasets:
            click.echo(f"No local datasets found in {local_data_dir}")
            return
        for dataset_name in local_datasets:
            folds = discover_folds(local_data_dir, dataset_name)
            click.echo(f"{dataset_name} ({len(folds)} folds)")
        return

    model_names = expand_csv(models)
    if not model_names:
        raise click.UsageError("Pass at least one model with --models.")

    datasets = expand_csv(dataset_names)
    if not datasets:
        if not local_datasets:
            raise click.UsageError(
                "No datasets were provided and no local data directory was found. "
                "Pass --datasets, for example: --datasets Crop"
            )
        datasets = local_datasets

    click.echo(f"Models: {', '.join(model_names)}")
    click.echo(f"Datasets: {', '.join(datasets)}")
    cache = make_cache(storage, output_dir, s3_uri)
    metadata = run_metadata(n_jobs=n_jobs)
    click.echo(f"Results: {s3_uri if storage == 's3' else output_dir}")

    for dataset_name in datasets:
        if fold_spec == "all":
            folds = discover_folds(local_data_dir, dataset_name)
            if not folds:
                folds = list(range(30))
        else:
            folds = [int(part.strip()) for part in fold_spec.split(",") if part.strip()]

        for fold in folds:
            for model_name in model_names:
                filename = result_filename(dataset_name, model_name, fold)
                if cache.exists(filename) and not overwrite:
                    click.echo(f"Skipping existing: {filename}")
                    continue

                stats = {
                    "dataset": dataset_name,
                    "fold": fold,
                    "model": model_name,
                    "random_state": fold,
                    "status": "ok",
                    **metadata,
                }

                click.echo(f"Running dataset={dataset_name} fold={fold} model={model_name}")
                model = None
                try:
                    model = get_model(model_name, random_state=fold, n_jobs=n_jobs)
                except ValueError as exc:
                    raise click.UsageError(str(exc)) from exc

                try:
                    X_train, y_train, X_test, y_test = load_benchmark_fold(
                        local_data_dir, dataset_name, fold
                    )
                    stats.update(
                        {
                            "n_train": len(y_train),
                            "n_test": len(y_test),
                            "n_classes": int(len(np.unique(y_train))),
                        }
                    )

                    t0 = perf_counter()
                    model.fit(X_train, y_train)
                    fit_seconds = perf_counter() - t0
                    stats["fit_seconds"] = fit_seconds
                    stats["fit_seconds_per_sample"] = fit_seconds / len(y_train)

                    t0 = perf_counter()
                    preds = model.predict(X_test)
                    predict_seconds = perf_counter() - t0
                    stats["predict_seconds"] = predict_seconds
                    stats["inference_seconds"] = predict_seconds
                    stats["predict_seconds_per_sample"] = predict_seconds / len(y_test)
                    stats["inference_seconds_per_sample"] = predict_seconds / len(y_test)
                    stats["total_seconds"] = fit_seconds + predict_seconds
                    stats["test_accuracy"] = float(accuracy_score(y_test, preds))
                except Exception as exc:
                    stats["status"] = "error"
                    stats["error"] = repr(exc)
                    click.echo(f"Error: {exc}", err=True)
                finally:
                    if model is not None and hasattr(model, "cleanup"):
                        model.cleanup()

                cache.add(pl.DataFrame([stats]), filename)


if __name__ == "__main__":
    main()
