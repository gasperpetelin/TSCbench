"""Run a TSC benchmark using tsml-format result files.

Examples:
    uv run python scripts/run_benchmark2.py --classifiers ROCKET,Catch22 --folds 0
    uv run python scripts/run_benchmark2.py --classifiers TSCGlue-Accuracy-GPU --datasets Crop --folds 0,1,2
    uv run python scripts/run_benchmark2.py --evaluate-only
"""

import os
import random
import sys
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
    "TSCGlueWeaselV2-Accuracy-GPU",
    "TSCGlueWeaselV2-LogLoss-GPU",
    "TSCGlueWeaselV2-ROCAUC-GPU",
    "TSCGlueDual-Accuracy-GPU",
    "TSCGlueDual-LogLoss-GPU",
    "TSCGlueDual-ROCAUC-GPU",
    "TSCGlueMean-GPU",
    "TSCGlueMeanV2-GPU",
    "TSCGlueMeanBalanced-GPU",
    "TSCGlueET-GPU",
    "TSCGlueETAll-GPU",
    "TSCGlueETAllV2-GPU",
    "TSCGlueRidgeAll-GPU",
    # TSCGlueEnhanced — one class, three presets (low/medium/high)
    "TSCGlueEnhanced-Low-LogLoss-GPU",
    "TSCGlueEnhanced-Medium-LogLoss-GPU",
    "TSCGlueEnhanced-High-LogLoss-GPU",
    # TSCGlueEnhancedV2 — full preset x eval_metric grid (served head depends on both)
    "TSCGlueEnhancedV2-Low-Accuracy-GPU",
    "TSCGlueEnhancedV2-Low-F1-GPU",
    "TSCGlueEnhancedV2-Low-ROCAUC-GPU",
    "TSCGlueEnhancedV2-Low-LogLoss-GPU",
    "TSCGlueEnhancedV2-Medium-Accuracy-GPU",
    "TSCGlueEnhancedV2-Medium-F1-GPU",
    "TSCGlueEnhancedV2-Medium-ROCAUC-GPU",
    "TSCGlueEnhancedV2-Medium-LogLoss-GPU",
    "TSCGlueEnhancedV2-High-Accuracy-GPU",
    "TSCGlueEnhancedV2-High-F1-GPU",
    "TSCGlueEnhancedV2-High-ROCAUC-GPU",
    "TSCGlueEnhancedV2-High-LogLoss-GPU",
    # tscglue.fallback feature-pipeline baselines (fallback candidates)
    "f-QuantET",
    "f-MultiET",
    "f-MRHydraET",
    "f-ShapeDictET",
    "f-AllFeaturesET",
    "f-MRHydraLogistic",
    "f-MRHydraRidge",
    "f-AllFeaturesRidge",
]


def make_classifier(name: str, random_state: int, n_jobs: int):
    from tscglue.fallback import BASELINES
    from tscglue.models import (
        TSCGlueClassifier,
        TSCGlueDual,
        TSCGlueEnhanced,
        TSCGlueEnhancedV2,
        TSCGlueET,
        TSCGlueETAll,
        TSCGlueETAllV2,
        TSCGlueMean,
        TSCGlueMeanBalanced,
        TSCGlueMeanV2,
        TSCGlueRidgeAll,
        TSCGlueWeaselV2,
    )

    if name.startswith("f-") and name[2:] in BASELINES:
        return BASELINES[name[2:]](random_state=random_state, n_jobs=n_jobs, verbose=1)
    if name.startswith("TSCGlueEnhancedV2-") and name.endswith("-GPU"):
        # TSCGlueEnhancedV2-<Preset>-<Metric>-GPU, e.g. TSCGlueEnhancedV2-High-F1-GPU.
        # Unlike V1, every preset x metric pair is a distinct served head, so all
        # 12 combinations are worth running.
        import torch
        _preset, _metric = name[len("TSCGlueEnhancedV2-"):-len("-GPU")].split("-")
        _metric_map = {"Accuracy": "accuracy", "F1": "f1", "LogLoss": "log_loss", "ROCAUC": "roc_auc"}
        return TSCGlueEnhancedV2(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(),
            eval_metric=_metric_map[_metric], preset=_preset.lower(),
        )
    if name.startswith("TSCGlueEnhanced-") and name.endswith("-GPU"):
        # TSCGlueEnhanced-<Preset>-<Metric>-GPU, e.g. TSCGlueEnhanced-Low-LogLoss-GPU
        import torch
        _preset, _metric = name[len("TSCGlueEnhanced-"):-len("-GPU")].split("-")
        _metric_map = {"Accuracy": "accuracy", "LogLoss": "log_loss", "ROCAUC": "roc_auc"}
        return TSCGlueEnhanced(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(),
            eval_metric=_metric_map[_metric], preset=_preset.lower(),
        )
    if name == "TSCGlue-Accuracy-GPU":
        import torch
        return TSCGlueClassifier(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(), eval_metric="accuracy",
        )
    if name == "TSCGlue-LogLoss-GPU":
        import torch
        return TSCGlueClassifier(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(), eval_metric="log_loss",
        )
    if name == "TSCGlue-ROCAUC-GPU":
        import torch
        return TSCGlueClassifier(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(), eval_metric="roc_auc",
        )
    if name == "TSCGlueWeaselV2-Accuracy-GPU":
        import torch
        return TSCGlueWeaselV2(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(), eval_metric="accuracy",
        )
    if name == "TSCGlueWeaselV2-LogLoss-GPU":
        import torch
        return TSCGlueWeaselV2(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(), eval_metric="log_loss",
        )
    if name == "TSCGlueWeaselV2-ROCAUC-GPU":
        import torch
        return TSCGlueWeaselV2(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(), eval_metric="roc_auc",
        )
    if name == "TSCGlueDual-Accuracy-GPU":
        import torch
        return TSCGlueDual(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(), eval_metric="accuracy",
        )
    if name == "TSCGlueDual-LogLoss-GPU":
        import torch
        return TSCGlueDual(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(), eval_metric="log_loss",
        )
    if name == "TSCGlueDual-ROCAUC-GPU":
        import torch
        return TSCGlueDual(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(), eval_metric="roc_auc",
        )
    if name == "TSCGlueMean-GPU":
        import torch
        return TSCGlueMean(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(),
        )
    if name == "TSCGlueMeanV2-GPU":
        import torch
        return TSCGlueMeanV2(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(),
        )
    if name == "TSCGlueMeanBalanced-GPU":
        import torch
        return TSCGlueMeanBalanced(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(),
        )
    if name == "TSCGlueET-GPU":
        import torch
        return TSCGlueET(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(),
        )
    if name == "TSCGlueETAll-GPU":
        import torch
        return TSCGlueETAll(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(),
        )
    if name == "TSCGlueETAllV2-GPU":
        import torch
        return TSCGlueETAllV2(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(),
        )
    if name == "TSCGlueRidgeAll-GPU":
        import torch
        return TSCGlueRidgeAll(
            verbose=10, random_state=random_state, n_jobs=n_jobs,
            n_gpus=torch.cuda.device_count(),
        )
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
@click.option(
    "--results-dir",
    default="generated_results",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory for summary CSVs and critical-difference diagrams.",
)
@click.option("-j", "--n-jobs", default=8, show_default=True, type=int)
@click.option("--overwrite", is_flag=True, help="Re-run and overwrite existing results.")
@click.option("--evaluate", is_flag=True, help="Run evaluation after benchmarking.")
@click.option(
    "--evaluate-only",
    is_flag=True,
    help="Skip benchmarking; only run evaluation on existing results.",
)
def main(
    classifiers,
    datasets,
    folds,
    data_dir,
    output_dir,
    results_dir,
    n_jobs,
    overwrite,
    evaluate,
    evaluate_only,
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

    if not evaluate_only:
        combos = list(product(classifier_names, dataset_list, fold_ids))
        random.shuffle(combos)
        click.echo(f"\nRunning {len(combos)} experiments...\n")

        for i, (clf_name, dataset, r) in enumerate(combos, start=1):
            try:
                clf = make_classifier(clf_name, random_state=r, n_jobs=n_jobs)
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
                click.echo(
                    f"ERROR {clf_name} {dataset} resample={r}: {exc}", err=True
                )

    if evaluate or evaluate_only:
        from tsml_eval.evaluation import evaluate_classifiers_by_problem

        click.echo(f"\nEvaluating results -> {results_dir}")
        evaluate_classifiers_by_problem(
            str(output_dir),
            classifier_names,
            dataset_list,
            str(results_dir),
            resamples=len(fold_ids),
        )


if __name__ == "__main__":
    main()
