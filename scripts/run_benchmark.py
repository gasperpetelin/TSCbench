"""Run a TSC benchmark with Rocket, MiniRocket, Catch22, and TSCGlue.

Examples:
    uv run python scripts/run_benchmark.py --datasets Crop --models tscglue --folds 0
    uv run python scripts/run_benchmark.py --datasets ArrowHead --models rocket,catch22
"""

# ruff: noqa: E402

import os
import sys
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
from tscglue.data_loader import DATA_DIR as TSCGLUE_DATA_DIR
from tscglue.models import TSCGlue
from tscbench.utils import LocalFileCache, LogsFileCache, S3FileCache, discover_datasets, hardware_info, load_ucr_fold, software_versions


def run_metadata(n_jobs: int) -> dict:
    return {
        "n_jobs": n_jobs,
        "hardware": hardware_info(),
        "versions": software_versions(),
    }


def get_model(model_name: str, random_state: int, n_jobs: int):
    if model_name == "rocket":
        return RocketClassifier(random_state=random_state, n_jobs=n_jobs)
    if model_name == "minirocket":
        return MiniRocketClassifier(random_state=random_state, n_jobs=n_jobs)
    if model_name == "catch22":
        return Catch22Classifier(random_state=random_state, n_jobs=n_jobs)
    if model_name == "tscglue":
        return TSCGlue(random_state=random_state, n_jobs=n_jobs)
    raise ValueError(f"Unknown model name: {model_name}")



def make_cache(storage: str, output_dir: Path, s3_uri: str):
    if storage == "s3":
        return S3FileCache(s3_uri)
    if storage == "logs":
        return LogsFileCache()
    return LocalFileCache(output_dir)


@click.command()
@click.option("-m", "--models", multiple=True, required=True, help="Models to run. May be repeated or comma-separated.")
@click.option("-d", "--datasets", "dataset_names", multiple=True, help="Datasets to run. May be repeated or comma-separated.")
@click.option("-f", "--folds", "fold_spec", default="0", show_default=True, help="Folds to run, comma-separated, or 'all'.")
@click.option("--data-dir", default="data", show_default=True, type=click.Path(path_type=Path))
@click.option("-o", "--output-dir", default="artifacts/results", show_default=True, type=click.Path(path_type=Path))
@click.option("--storage", type=click.Choice(["disk", "s3", "logs"]), default="logs", show_default=True)
@click.option("--s3-uri", default="s3://tsc-bench/performance-benchmarking", show_default=True)
@click.option("-j", "--n-jobs", default=8, show_default=True, type=int)
@click.option("--overwrite", is_flag=True)
@click.option("--list-datasets", is_flag=True)
def main(models, dataset_names, fold_spec, data_dir, output_dir, storage, s3_uri, n_jobs, overwrite, list_datasets):
    local_data_dir = data_dir if data_dir.exists() else Path(TSCGLUE_DATA_DIR)
    local_datasets = discover_datasets(local_data_dir)

    if list_datasets:
        for name in local_datasets:
            click.echo(f"{name} (30 folds)")
        return

    model_names = [m.strip() for ms in models for m in ms.split(",") if m.strip()]
    datasets = [d.strip() for ds in dataset_names for d in ds.split(",") if d.strip()] or local_datasets

    if not datasets:
        raise click.UsageError("No datasets found. Pass --datasets or add data to --data-dir.")

    cache = make_cache(storage, output_dir, s3_uri)
    metadata = run_metadata(n_jobs=n_jobs)

    click.echo(f"Models:   {', '.join(model_names)}")
    click.echo(f"Datasets: {', '.join(datasets)}")
    click.echo(f"Results:  {s3_uri if storage == 's3' else output_dir}")

    for dataset_name in datasets:
        if fold_spec == "all":
            folds = list(range(30))
        else:
            folds = [int(f.strip()) for f in fold_spec.split(",") if f.strip()]

        for model_name, fold in product(model_names, folds):
            model = None
            try:
                model = get_model(model_name, random_state=fold, n_jobs=n_jobs)
                model_params = {k: str(v) for k, v in model.get_params().items()}

                stats = {
                    "dataset": dataset_name,
                    "fold": fold,
                    "model": model_name,
                    "random_state": fold,
                    **metadata,
                    "model_params": model_params,
                }

                filename = f"{pl.DataFrame([{k: stats[k] for k in ('dataset', 'fold', 'model', 'random_state', 'n_jobs', 'hardware', 'versions', 'model_params')}]).hash_rows(seed=42, seed_1=1, seed_2=2, seed_3=3).item()}.parquet"

                if cache.exists(filename) and not overwrite:
                    click.echo(f"Skipping: dataset={dataset_name} fold={fold} model={model_name}")
                    continue

                click.echo(f"Running:  dataset={dataset_name} fold={fold} model={model_name}")

                X_train, y_train, X_test, y_test = load_ucr_fold(local_data_dir, dataset_name, fold)
                stats["dataset_stats"] = {
                    "n_train":     len(y_train),
                    "n_test":      len(y_test),
                    "n_classes":   int(len(np.unique(y_train))),
                    "n_channels":  int(X_train.shape[1]),
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
                click.echo(f"Error: dataset={dataset_name} fold={fold} model={model_name}: {exc}", err=True)
            finally:
                if model is not None and hasattr(model, "cleanup"):
                    model.cleanup()


if __name__ == "__main__":
    main()
