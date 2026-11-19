"""Benchmark TSCGlue PR #23 (best preset serves the wmean-T-ll-L2 blend) against current main.

Each version gets its own env next to .venv (same Python and packages, other tscglue):
  old: main at 80a54c0, fits High-Accuracy, High-LogLoss and Best-LogLoss (meta-probability-et)
  new: PR #23 at bc444fb, fits Best-LogLoss (the blend; it is served for every eval_metric)

    # build both envs: clone TSCGlue into .blend_env/<version>, create and verify .venv-blend-<version>
    uv run --no-sync python scripts/run_blend_head.py setup-env
    # queue the jobs: 50 smallest UCR datasets, folds 0-9, one tsp job per (dataset, fold, version)
    uv run --no-sync python scripts/run_blend_head.py submit --smallest 50 -f 0-9
    # what one job runs
    .venv-blend-new/bin/python scripts/run_blend_head.py run -v new -d Beef -f 3

Results: blend_results/TSCGlueEnhancedV4-<Preset>-<Metric>-GPU-<Version>/Predictions/<dataset>/testResample<fold>.csv
"""

import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "blend_results"
TSCGLUE_REPO = "https://github.com/gasperpetelin/TSCGlue"
VERSIONS = {
    "old": {"commit": "80a54c0", "served_best": "meta-probability-et",
            "classifiers": ["High-Accuracy", "High-LogLoss", "Best-LogLoss"]},
    "new": {"commit": "bc444fb", "served_best": "wmean-T-ll-L2", "classifiers": ["Best-LogLoss"]},
}


def env_dir(version):
    return PROJECT_ROOT / f".venv-blend-{version}"


def check_env(version):
    return (
        "import tscglue, tscglue.models as m; "
        f"assert m._ENHANCED_SERVED_HEAD[('best', 'log_loss')] == {VERSIONS[version]['served_best']!r}, "
        "tscglue.__file__; "
        "import torch, tsml_eval, aeon; print(tscglue.__file__, 'cuda', torch.cuda.is_available())"
    )


def classifier_name(base, version):
    return f"TSCGlueEnhancedV4-{base}-GPU-{version.capitalize()}"


def result_path(output_dir, base, version, dataset, fold):
    return (Path(output_dir) / classifier_name(base, version) / "Predictions" / dataset
            / f"testResample{fold}.csv")


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


def smallest_datasets(n, data_dir=DATA_DIR):
    from aeon.datasets.tsc_datasets import univariate_equal_length

    return sorted(univariate_equal_length, key=lambda d: (n_train(d, data_dir), d))[:n]


@click.group()
def cli():
    pass


@cli.command()
@click.option("-v", "--version", required=True, type=click.Choice(list(VERSIONS)))
@click.option("-d", "--dataset", required=True)
@click.option("-f", "--fold", required=True, type=int)
@click.option("-j", "--n-jobs", default=8, show_default=True, type=int)
@click.option("-o", "--output-dir", default=str(OUTPUT_DIR), show_default=True)
@click.option("--data-dir", default=str(DATA_DIR), show_default=True)
@click.option("--overwrite", is_flag=True, help="Re-run and overwrite existing results.")
def run(version, dataset, fold, n_jobs, output_dir, data_dir, overwrite):
    """Fit every classifier of one version on one dataset and fold."""
    import run_benchmark2
    from tsml_eval.experiments import load_and_run_classification_experiment

    failed = False
    for base in VERSIONS[version]["classifiers"]:
        name = classifier_name(base, version)
        try:
            # Everything the model writes goes under run_dir, removed after the experiment.
            with tempfile.TemporaryDirectory() as run_dir:
                clf = run_benchmark2.make_classifier(
                    f"TSCGlueEnhancedV4-{base}-GPU", random_state=fold, n_jobs=n_jobs, runs_dir=run_dir,
                )
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
@click.option("-d", "--datasets", default=None, help="Comma-separated datasets.")
@click.option("--smallest", type=int, default=None, help="The N UCR datasets with the fewest training cases.")
@click.option("-f", "--folds", default="0-9", show_default=True, help="Folds, e.g. 0-9 or 0,3,5.")
@click.option("-j", "--n-jobs", default=8, show_default=True, type=int)
@click.option("-o", "--output-dir", default=str(OUTPUT_DIR), show_default=True)
@click.option("--data-dir", default=str(DATA_DIR), show_default=True)
@click.option("--heavy", default="ElectricDevices,Crop", show_default=True,
              help="Datasets whose jobs take --heavy-slots tsp slots, so they run alone (OOM when concurrent).")
@click.option("--heavy-slots", default=6, show_default=True, type=int)
@click.option("--max-queued", default=900, show_default=True, type=int,
              help="Stop once this many blend- jobs are queued or running: tsp blocks at ~1000 waiting jobs.")
@click.option("--dry-run", is_flag=True, help="Print the tsp commands instead of queueing them.")
def submit(datasets, smallest, folds, n_jobs, output_dir, data_dir, heavy, heavy_slots, max_queued, dry_run):
    """Queue one tsp job per (dataset, fold, version), dataset by dataset."""
    for version in VERSIONS:
        check = subprocess.run([str(env_dir(version) / "bin" / "python"), "-c", check_env(version)],
                               capture_output=True, text=True)
        if check.returncode != 0:
            raise click.ClickException(f"{env_dir(version)} is missing or wrong; run setup-env.\n{check.stderr}")
    names = [d.strip() for d in datasets.split(",") if d.strip()] if datasets else smallest_datasets(smallest, data_dir)
    # labels of blend- jobs already queued or running, so they are not queued twice
    in_tsp = {line.split("[", 1)[1].split("]", 1)[0] for line in subprocess.run(
        ["tsp"], check=True, capture_output=True, text=True).stdout.splitlines()[1:]
        if line.split()[1] in ("queued", "running") and "[blend-" in line}
    queued = skipped = 0
    for dataset in names:
        for fold in parse_folds(folds):
            for version, spec in VERSIONS.items():
                if f"blend-{version}-{dataset}-f{fold}" in in_tsp or all(
                        result_path(output_dir, b, version, dataset, fold).exists() for b in spec["classifiers"]):
                    skipped += 1
                    continue
                if len(in_tsp) + queued >= max_queued:
                    click.echo(f"Queued {queued} jobs, stopped at --max-queued {max_queued}; submit again later")
                    return
                slots = ["-N", str(heavy_slots)] if dataset in heavy.split(",") else []
                cmd = [
                    "tsp", *slots, "-L", f"blend-{version}-{dataset}-f{fold}",
                    "env", "HF_HUB_OFFLINE=1", "CUDA_VISIBLE_DEVICES=0",
                    "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OMP_NUM_THREADS=1",
                    str(env_dir(version) / "bin" / "python"), str(Path(__file__).resolve()), "run",
                    "-v", version, "-d", dataset, "-f", str(fold), "-j", str(n_jobs),
                    "-o", str(output_dir), "--data-dir", str(data_dir),
                ]
                if dry_run:
                    click.echo(shlex.join(cmd))
                else:
                    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL)
                queued += 1
    click.echo(f"{'Would queue' if dry_run else 'Queued'} {queued} jobs, {skipped} already done")


@cli.command("setup-env")
def setup_env():
    """For each version: clone TSCGlue at its commit and build its env with .venv's packages."""
    main_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    freeze = subprocess.run(["uv", "pip", "freeze", "--python", main_python],
                            check=True, capture_output=True, text=True).stdout.splitlines()
    # .venv's exact versions, minus tscglue and this project (both installed editable below).
    reqs = [line for line in freeze if not line.startswith(("tscglue", "-e ", "tscbench"))]
    version_str = subprocess.run([main_python, "-c", "import platform; print(platform.python_version())"],
                                 check=True, capture_output=True, text=True).stdout.strip()
    for version, spec in VERSIONS.items():
        src = PROJECT_ROOT / ".blend_env" / version
        if not src.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", TSCGLUE_REPO, str(src)], check=True)
        subprocess.run(["git", "-C", str(src), "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", str(src), "checkout", spec["commit"]], check=True)
        env = env_dir(version)
        if not env.exists():
            subprocess.run(["uv", "venv", "--python", version_str, str(env)], check=True)
        env_python = str(env / "bin" / "python")
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
            f.write("\n".join(reqs) + "\n")
            f.flush()
            # --no-deps: .venv's freeze is not a consistent resolution (e.g. fsspec vs datasets).
            subprocess.run([
                "uv", "pip", "install", "--python", env_python, "--no-deps", "-r", f.name,
                "--extra-index-url", "https://download.pytorch.org/whl/cu132",
                "--index-strategy", "unsafe-best-match",
            ], check=True)
        subprocess.run(["uv", "pip", "install", "--python", env_python, "--no-deps",
                        "-e", str(src), "-e", str(PROJECT_ROOT)], check=True)
        subprocess.run([env_python, "-c", check_env(version)], check=True)


# tscglue starts workers with forkserver, which re-imports this module: keep the run behind the guard.
if __name__ == "__main__":
    cli()
