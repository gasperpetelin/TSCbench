"""Benchmark TSCGlue-H, accuracy and log-loss heads, with the current folds and with round-robin folds.

One tsp job per (dataset, fold, fold method). Each job fits TSCGlueEnhancedV4-High-Accuracy-GPU
and TSCGlueEnhancedV4-High-LogLoss-GPU, built by run_benchmark2.make_classifier.

The jobs need a tscglue with the fold_method parameter (TSCGlue PR #20, commit c81d784), installed
in its own env, .venv-fold-method, next to .venv (same Python and packages, other tscglue):

    # build the env: clone TSCGlue into .fold_method_env/TSCGlue, create and verify .venv-fold-method
    uv run --no-sync python scripts/run_fold_method.py setup-env
    # queue the jobs: 50 smallest UCR datasets, folds 0-9
    .venv-fold-method/bin/python scripts/run_fold_method.py submit --smallest 50 -f 0-9
    # queue the jobs: chosen datasets and folds
    .venv-fold-method/bin/python scripts/run_fold_method.py submit -d Beef,Rock -f 0,3,5
    # defaults: all 112 UCR equal-length datasets, folds 0-29
    .venv-fold-method/bin/python scripts/run_fold_method.py submit
    # what one job runs
    .venv-fold-method/bin/python scripts/run_fold_method.py run -d Beef -f 3 -m round-robin

Results: fold_method_results/<classifier>/Predictions/<dataset>/testResample<fold>.csv, with
<classifier> = TSCGlueEnhancedV4-High-{Accuracy,LogLoss}-GPU-{StratifiedKFold,RoundRobin}.

"stratified-kfold" is tscglue's current get_folds: StratifiedKFold, unstratified KFold when every
class has fewer cases than k_folds. "round-robin" uses stratified round-robin folds for every split.
A tscglue without fold_method raises an error.
"""

import inspect
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "fold_method_results"
ENV_DIR = PROJECT_ROOT / ".venv-fold-method"
TSCGLUE_DIR = PROJECT_ROOT / ".fold_method_env" / "TSCGlue"
TSCGLUE_REPO = "https://github.com/gasperpetelin/TSCGlue"
TSCGLUE_COMMIT = "c81d784"
CHECK_ENV = (
    "import inspect, tscglue, tscglue.models as m; "
    "assert 'fold_method' in inspect.signature(m.TSCGlueEnhancedV4.__init__).parameters, tscglue.__file__; "
    "import torch, tsml_eval, aeon; print(tscglue.__file__, 'cuda', torch.cuda.is_available())"
)
FOLD_METHODS = {"stratified-kfold": "StratifiedKFold", "round-robin": "RoundRobin"}
METRICS = ("Accuracy", "LogLoss")


def classifier_name(metric, fold_method):
    return f"TSCGlueEnhancedV4-High-{metric}-GPU-{FOLD_METHODS[fold_method]}"


def result_path(output_dir, metric, fold_method, dataset, fold):
    return Path(output_dir) / classifier_name(metric, fold_method) / "Predictions" / dataset / f"testResample{fold}.csv"


def parse_folds(folds):
    """'0-9' or '0,3,5' (or a mix, '0-4,7')."""
    out = []
    for part in folds.split(","):
        lo, _, hi = part.strip().partition("-")
        out.extend(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    return sorted(set(out))


def n_train(dataset, data_dir):
    with open(Path(data_dir) / dataset / f"{dataset}0_TRAIN.ts") as f:
        for line in f:
            if line.strip().lower() == "@data":
                break
        return sum(1 for line in f if line.strip())


def use_fold_method(clf, fold_method):
    """Set the fold method on clf; raise if the installed tscglue has no fold_method parameter."""
    import tscglue

    if "fold_method" not in inspect.signature(type(clf).__init__).parameters:
        raise RuntimeError(
            f"tscglue at {Path(tscglue.__file__).parent} has no fold_method parameter: "
            f"run with {ENV_DIR}/bin/python (built by setup-env)"
        )
    clf.set_params(fold_method=fold_method)


@click.group()
def cli():
    pass


@cli.command()
@click.option("-d", "--dataset", required=True)
@click.option("-f", "--fold", required=True, type=int)
@click.option("-m", "--fold-method", required=True, type=click.Choice(list(FOLD_METHODS)))
@click.option("-j", "--n-jobs", default=8, show_default=True, type=int)
@click.option("-o", "--output-dir", default=str(OUTPUT_DIR), show_default=True)
@click.option("--data-dir", default=str(DATA_DIR), show_default=True)
@click.option("--overwrite", is_flag=True, help="Re-run and overwrite existing results.")
def run(dataset, fold, fold_method, n_jobs, output_dir, data_dir, overwrite):
    """Fit High-Accuracy and High-LogLoss on one dataset and fold with one fold method."""
    import run_benchmark2
    from tsml_eval.experiments import load_and_run_classification_experiment

    failed = False
    for metric in METRICS:
        name = classifier_name(metric, fold_method)
        try:
            # Everything the model writes goes under run_dir, removed after the experiment.
            with tempfile.TemporaryDirectory() as run_dir:
                clf = run_benchmark2.make_classifier(
                    f"TSCGlueEnhancedV4-High-{metric}-GPU", random_state=fold, n_jobs=n_jobs,
                    runs_dir=run_dir,
                )
                use_fold_method(clf, fold_method)
                click.echo(f"{name}  {dataset}  resample={fold}")
                load_and_run_classification_experiment(
                    data_dir, output_dir, dataset, clf, classifier_name=name, resample_id=fold,
                    predefined_resample=True, overwrite=overwrite,
                )
        except Exception as exc:
            failed = True
            click.echo(f"ERROR {name} {dataset} resample={fold}: {exc}", err=True)
    sys.exit(1 if failed else 0)


@cli.command()
@click.option("-d", "--datasets", default=None, help="Comma-separated datasets. Default: all 112 UCR equal-length datasets.")
@click.option("-f", "--folds", default="0-29", show_default=True, help="Folds, e.g. 0-9 or 0,3,5.")
@click.option("-m", "--fold-methods", default=",".join(FOLD_METHODS), show_default=True)
@click.option("--smallest", type=int, default=None, help="Keep only the N datasets with the fewest training cases.")
@click.option("-j", "--n-jobs", default=8, show_default=True, type=int)
@click.option("-o", "--output-dir", default=str(OUTPUT_DIR), show_default=True)
@click.option("--data-dir", default=str(DATA_DIR), show_default=True)
@click.option("--overwrite", is_flag=True, help="Queue and overwrite even where results exist.")
@click.option("--dry-run", is_flag=True, help="Print the tsp commands instead of queueing them.")
def submit(datasets, folds, fold_methods, smallest, n_jobs, output_dir, data_dir, overwrite, dry_run):
    """Queue one tsp job per (dataset, fold, fold method)."""
    from aeon.datasets.tsc_datasets import univariate_equal_length

    check = subprocess.run([str(ENV_DIR / "bin" / "python"), "-c", CHECK_ENV], capture_output=True, text=True)
    if check.returncode != 0:
        raise click.ClickException(
            f"{ENV_DIR} is missing or its tscglue has no fold_method; run setup-env first.\n{check.stderr}"
        )
    names = [d.strip() for d in datasets.split(",") if d.strip()] if datasets else sorted(univariate_equal_length)
    if smallest:
        names = sorted(names, key=lambda d: (n_train(d, data_dir), d))[:smallest]
    methods = [m.strip() for m in fold_methods.split(",") if m.strip()]
    for m in methods:
        if m not in FOLD_METHODS:
            raise click.BadParameter(f"{m!r} is not one of {list(FOLD_METHODS)}", param_hint="--fold-methods")

    # labels of fm- jobs already queued or running, so they are not queued twice
    in_tsp = {line.split("[", 1)[1].split("]", 1)[0] for line in subprocess.run(
        ["tsp"], check=True, capture_output=True, text=True).stdout.splitlines()[1:]
        if line.split()[1] in ("queued", "running") and "[fm-" in line}

    queued = skipped = 0
    # dataset by dataset: all folds of one dataset are queued before the next dataset
    for dataset in names:
        for fold in parse_folds(folds):
            for method in methods:
                label = f"fm-{dataset}-f{fold}-{method}"
                if label in in_tsp or (not overwrite and all(
                        result_path(output_dir, m, method, dataset, fold).exists() for m in METRICS)):
                    skipped += 1
                    continue
                cmd = [
                    "tsp", "-L", label,
                    "env", "HF_HUB_OFFLINE=1", "CUDA_VISIBLE_DEVICES=0",
                    "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OMP_NUM_THREADS=1",
                    str(ENV_DIR / "bin" / "python"), str(Path(__file__).resolve()), "run",
                    "-d", dataset, "-f", str(fold), "-m", method, "-j", str(n_jobs),
                    "-o", str(output_dir), "--data-dir", str(data_dir),
                ] + (["--overwrite"] if overwrite else [])
                if dry_run:
                    click.echo(shlex.join(cmd))
                else:
                    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL)
                queued += 1
    click.echo(f"{'Would queue' if dry_run else 'Queued'} {queued} jobs ({len(names)} datasets x "
               f"{len(parse_folds(folds))} folds x {len(methods)} fold methods, {skipped} already done or in tsp)")
    click.echo(f"Results: {output_dir}/<classifier>/Predictions/<dataset>/testResample<fold>.csv")


@cli.command("setup-env")
def setup_env():
    """Clone TSCGlue at TSCGLUE_COMMIT and build ENV_DIR: .venv's Python and packages, that tscglue."""
    if not TSCGLUE_DIR.exists():
        TSCGLUE_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", TSCGLUE_REPO, str(TSCGLUE_DIR)], check=True)
    subprocess.run(["git", "-C", str(TSCGLUE_DIR), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(TSCGLUE_DIR), "checkout", TSCGLUE_COMMIT], check=True)

    main_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    if not ENV_DIR.exists():
        version = subprocess.run([main_python, "-c", "import platform; print(platform.python_version())"],
                                 check=True, capture_output=True, text=True).stdout.strip()
        subprocess.run(["uv", "venv", "--python", version, str(ENV_DIR)], check=True)
    env_python = str(ENV_DIR / "bin" / "python")
    freeze = subprocess.run(["uv", "pip", "freeze", "--python", main_python],
                            check=True, capture_output=True, text=True).stdout.splitlines()
    # .venv's exact versions, minus tscglue and this project (both installed editable below).
    reqs = [line for line in freeze if not line.startswith(("tscglue", "-e ", "tscbench"))]
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
        f.write("\n".join(reqs) + "\n")
        f.flush()
        # --no-deps: .venv's freeze is not a consistent resolution (e.g. fsspec vs datasets).
        subprocess.run([
            "uv", "pip", "install", "--python", env_python, "--no-deps", "-r", f.name,
            "--extra-index-url", "https://download.pytorch.org/whl/cu132", "--index-strategy", "unsafe-best-match",
        ], check=True)
    subprocess.run(["uv", "pip", "install", "--python", env_python, "--no-deps",
                    "-e", str(TSCGLUE_DIR), "-e", str(PROJECT_ROOT)], check=True)
    subprocess.run([env_python, "-c", CHECK_ENV], check=True)


# tscglue starts workers with forkserver, which re-imports this module: keep the run behind the guard.
if __name__ == "__main__":
    cli()
