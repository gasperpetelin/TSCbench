"""Run a TSC benchmark using tsml-format result files.

Examples:
    uv run python scripts/run_benchmark2.py --classifiers ROCKET,Catch22 --folds 0
    uv run python scripts/run_benchmark2.py --classifiers TSCGlue-Accuracy-GPU --datasets Crop --folds 0,1,2
"""

import os
import random
import sys
import tempfile
from itertools import product
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import aeon.base._base_collection as _base_collection
import click
from aeon.classification.hybrid import HIVECOTEV2
from aeon.datasets.tsc_datasets import univariate_equal_length
from tsml_eval.experiments import load_and_run_classification_experiment
from tsml_eval.publications.y2023.tsc_bakeoff.set_bakeoff_classifier import (
    _set_bakeoff_classifier,
)

# aeon's zero-variance guard (aeon/base/_base_collection.py) runs on every recursive
# call an estimator makes internally, including short interval/shapelet slices that
# are legitimately near-constant up to float rounding (aeon#3570, aeon#3599). This
# false-positives HIVECOTEV2 out of otherwise-valid fits. No fixed threshold is safe:
# a 2-3 point window drawn from repeated raw readings can land at std ~1e-17 (machine
# epsilon), arbitrarily below any tolerance we pick. aeon's own numeric code already
# treats anything under AEON_NUMBA_STD_THRESHOLD (1e-8) as safe-to-treat-as-constant,
# so match the upstream fix (aeon PR #3598): demote the check to a warning instead.
_original_check_collection_variance = _base_collection.check_collection_variance


def _lenient_check_collection_variance(X, threshold=1e-7, raise_error=True):
    return _original_check_collection_variance(X, threshold=threshold, raise_error=False)


_base_collection.check_collection_variance = _lenient_check_collection_variance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_DEFAULT_DATA = str(PROJECT_ROOT / "data")

AVAILABLE_CLASSIFIERS = [
    "ROCKET",
    "MiniRocket",
    "MultiRocketHydra",
    "Catch22",
    "HIVECOTEV2",
    "TSCGlue-Accuracy-GPU",
    "TSCGlue-LogLoss-GPU",
    "TSCGlue-ROCAUC-GPU",
    # TSCGlueEnhancedV4 — full preset x eval_metric grid is accepted (F1/ROCAUC
    # included), but only the two headline metrics are benchmarked.
    "TSCGlueEnhancedV4-Low-Accuracy-GPU",
    "TSCGlueEnhancedV4-Low-LogLoss-GPU",
    "TSCGlueEnhancedV4-Medium-Accuracy-GPU",
    "TSCGlueEnhancedV4-Medium-LogLoss-GPU",
    "TSCGlueEnhancedV4-High-Accuracy-GPU",
    "TSCGlueEnhancedV4-High-LogLoss-GPU",
]


def make_classifier(name: str, random_state: int, n_jobs: int, runs_dir=None):
    # tscglue models write fold-model pickles and feature caches under
    # `<runs_dir>/<run_id>/`, defaulting to ./tscglue_runs. Only the feature caches
    # are cleaned up on their own; the pickles are not. `runs_dir` points them at a
    # per-experiment temp dir so the caller can delete the lot in one go.
    from tscglue.models import TSCGlueClassifier, TSCGlueEnhancedV4

    if name.startswith("TSCGlueEnhancedV4-") and name.endswith("-GPU"):
        # TSCGlueEnhancedV4-<Preset>-<Metric>-GPU, e.g. TSCGlueEnhancedV4-High-LogLoss-GPU.
        import torch
        _preset, _metric = name[len("TSCGlueEnhancedV4-"):-len("-GPU")].split("-")
        _metric_map = {"Accuracy": "accuracy", "F1": "f1", "LogLoss": "log_loss", "ROCAUC": "roc_auc"}
        return TSCGlueEnhancedV4(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(),
            eval_metric=_metric_map[_metric], preset=_preset.lower(),
            runs_dir=runs_dir,
            # Cap cases in flight at predict time so peak RAM and run-dir disk
            # scale with the batch rather than with len(X).
            predict_batch_size=1000,
        )
    if name == "TSCGlue-Accuracy-GPU":
        import torch
        return TSCGlueClassifier(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(), eval_metric="accuracy",
            runs_dir=runs_dir,
        )
    if name == "TSCGlue-LogLoss-GPU":
        import torch
        return TSCGlueClassifier(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(), eval_metric="log_loss",
            runs_dir=runs_dir,
        )
    if name == "TSCGlue-ROCAUC-GPU":
        import torch
        return TSCGlueClassifier(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(), eval_metric="roc_auc",
            runs_dir=runs_dir,
        )
    if name == "HIVECOTEV2":
        # DrCIF is the only HC2 component that does not ask joblib for threads, so at
        # n_jobs > 1 it fans out over loky processes. Those are fresh interpreters: they
        # re-import aeon and lose the variance patch above, and the strict check then
        # kills the fit from inside a worker. Pin DrCIF to threads so the ensemble stays
        # in one process -- STC/Arsenal/TDE already pass prefer="threads". n_estimators
        # has to be restated: HC2 replaces its default drcif_params dict, not merges it.
        return HIVECOTEV2(
            random_state=random_state,
            n_jobs=n_jobs,
            drcif_params={
                "n_estimators": HIVECOTEV2._DEFAULT_N_TREES,
                "parallel_backend": "threading",
            },
        )
    # Bakeoff classifiers keep no run dir of their own, so runs_dir does not apply.
    return _set_bakeoff_classifier(name, random_state=random_state, n_jobs=n_jobs)


@click.command()
@click.option(
    "-c",
    "--classifiers",
    default=",".join(AVAILABLE_CLASSIFIERS),
    show_default=True,
    help="Comma-separated list of classifiers to run.",
)
@click.option(
    "-d",
    "--datasets",
    default=None,
    help="Comma-separated list of datasets. Defaults to all 112 UCR equal-length datasets.",
)
@click.option(
    "-f",
    "--folds",
    default="all",
    show_default=True,
    help="Comma-separated fold IDs, or 'all' for 0-29.",
)
@click.option(
    "--data-dir",
    default=_DEFAULT_DATA,
    show_default=True,
    type=click.Path(),
    help="Root directory containing dataset fold files.",
)
@click.option(
    "-o",
    "--output-dir",
    default="model_results",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory for tsml-format prediction files.",
)
@click.option("-j", "--n-jobs", default=8, show_default=True, type=int)
@click.option("--overwrite", is_flag=True, help="Re-run and overwrite existing results.")
def main(
    classifiers,
    datasets,
    folds,
    data_dir,
    output_dir,
    n_jobs,
    overwrite,
):
    classifier_names = [c.strip() for c in classifiers.split(",") if c.strip()]
    dataset_list = (
        [d.strip() for d in datasets.split(",") if d.strip()]
        if datasets
        else sorted(univariate_equal_length)
    )

    if folds == "all":
        fold_ids = list(range(30))
    else:
        fold_ids = [int(f.strip()) for f in folds.split(",") if f.strip()]

    click.echo(f"Classifiers: {', '.join(classifier_names)}")
    click.echo(f"Datasets:    {len(dataset_list)} ({'all UCR' if not datasets else ', '.join(dataset_list)})")
    click.echo(f"Folds:       {fold_ids}")
    click.echo(f"Data dir:    {data_dir}")
    click.echo(f"Output dir:  {output_dir}")

    combos = list(product(classifier_names, dataset_list, fold_ids))
    random.shuffle(combos)
    click.echo(f"\nRunning {len(combos)} experiments...\n")

    for i, (clf_name, dataset, r) in enumerate(combos, start=1):
        try:
            # Everything the model writes goes under this dir, so it is gone by the
            # time the next experiment starts -- crashed runs included.
            with tempfile.TemporaryDirectory() as run_dir:
                clf = make_classifier(
                    clf_name, random_state=r, n_jobs=n_jobs, runs_dir=run_dir
                )
                click.echo(f"[{i}/{len(combos)}] {clf_name}  {dataset}  resample={r}")
                click.echo(f"    {clf!r}")
                load_and_run_classification_experiment(
                    data_dir,
                    str(output_dir),
                    dataset,
                    clf,
                    classifier_name=clf_name,
                    resample_id=r,
                    predefined_resample=True,
                    overwrite=overwrite,
                )
        except Exception as exc:
            click.echo(f"ERROR {clf_name} {dataset} resample={r}: {exc}", err=True)


if __name__ == "__main__":
    main()
