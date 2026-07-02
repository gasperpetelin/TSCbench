"""Run a TSC benchmark using tsml-format result files.

Examples:
    uv run python scripts/run_benchmark2.py --classifiers ROCKET,Catch22 --folds 0
    uv run python scripts/run_benchmark2.py --classifiers TSCGlue-Accuracy-GPU --datasets Crop --folds 0,1,2
    uv run python scripts/run_benchmark2.py --evaluate-only
"""

import random
import sys
from itertools import product
from pathlib import Path

import click
from aeon.classification.hybrid import HIVECOTEV2
from aeon.datasets.tsc_datasets import univariate_equal_length
from tsml_eval.experiments import load_and_run_classification_experiment
from tsml_eval.publications.y2023.tsc_bakeoff.set_bakeoff_classifier import (
    _set_bakeoff_classifier,
)

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
]


def make_classifier(name: str, random_state: int, n_jobs: int):
    from tscglue.models import TSCGlueClassifier

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

        for clf_name, dataset, r in combos:
            try:
                click.echo(f"{clf_name}  {dataset}  resample={r}")
                load_and_run_classification_experiment(
                    data_dir,
                    str(output_dir),
                    dataset,
                    make_classifier(clf_name, random_state=r, n_jobs=n_jobs),
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
