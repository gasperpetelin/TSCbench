"""tscbench-run: run one model, at any repo commit or package version, in a throwaway environment.

Builds a temporary venv with the package and its own dependencies (as that commit's or version's
pyproject asks for), runs every (dataset, resample) as its own process in it, writes tsml results
locally, records what ran in run_info.txt, and deletes the venv at the end.

    # TSCGlue at a branch, tag or commit (resolved to the full commit)
    uv run --no-sync tscbench-run --repo https://github.com/gasperpetelin/TSCGlue --ref main \\
        --extra torch --model tscglue.models:TSCGlueEnhancedV4 \\
        -p preset=best -p eval_metric=log_loss -p n_jobs=8 -p n_gpus=1 \\
        -d ucr:Beef,ucr:Coffee -r 0-9

    # an aeon model at a given aeon version
    uv run --no-sync tscbench-run --package aeon==1.6.0 \\
        --model aeon.classification.convolution_based:MultiRocketHydraClassifier -d ucr:Beef -r 0-4

Results: <output-dir>/<name>/Predictions/<dataset>/testResample<k>.csv, run_info.txt and one log
per run in <output-dir>/<name>/. Existing results are skipped unless --overwrite.
<output-dir> can be s3://bucket/prefix: each run is written to a local temp dir and uploaded
(results file and log) when it ends; AWS credentials and AWS_ENDPOINT_URL come from the environment.
Runs go in random order and each one is locked while it runs, so several copies of the same
command can work through the runs together (with S3, only copies on the same machine share locks;
copies elsewhere only skip runs whose results are already uploaded).
The model gets random_state=<resample> and runs_dir=<temp dir> when its __init__ takes them;
everything else (n_jobs, n_gpus, ...) comes from -p. Values are Python literals (True, 8, 0.5,
"x"); anything else is a string.
"""

import argparse
import ast
import fcntl
import hashlib
import importlib
import inspect
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = {"ucr": PROJECT_ROOT / "data"}
TSML_EVAL = "tsml-eval==0.7.0"  # only its results writer is used (numpy only), so no dependencies
PYTHON = "3.13"


# ----------------------------- inside the run's venv -----------------------------


def parse_params(pairs):
    params = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"parameter {pair!r} is not key=value")
        try:
            params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            params[key] = value
    return params


def run_one(args):
    """One (dataset, resample): fit, predict and write the tsml TEST results file.

    Only aeon's .ts loader and tsml-eval's results writer are used, so it works with whatever
    aeon version the package under test brings.
    """
    import time

    import numpy as np
    from aeon.datasets import load_from_ts_file
    from tsml_eval.utils.results_writing import write_classification_results

    # aeon < 1.6 raises on near-constant slices of valid data; 1.6 warns (aeon#3598). Match 1.6.
    try:
        import aeon.base._base_collection as base_collection

        original = base_collection.check_collection_variance
        base_collection.check_collection_variance = lambda X, threshold=1e-7, raise_error=True: (
            original(X, threshold=threshold, raise_error=False)
        )
    except (ImportError, AttributeError):
        pass

    module, _, cls_name = args.model.partition(":")
    cls = getattr(importlib.import_module(module), cls_name)
    folder = Path(args.data_dir) / args.dataset
    X_train, y_train = load_from_ts_file(str(folder / f"{args.dataset}{args.resample}_TRAIN.ts"))
    X_test, y_test = load_from_ts_file(str(folder / f"{args.dataset}{args.resample}_TEST.ts"))
    labels = np.unique(y_train)  # the same encoding as tsml-eval's LabelEncoder
    y_train, y_test = np.searchsorted(labels, y_train), np.searchsorted(labels, y_test)

    with tempfile.TemporaryDirectory() as run_dir:
        accepted = inspect.signature(cls.__init__).parameters
        defaults = {"random_state": args.resample, "runs_dir": run_dir}
        params = {k: v for k, v in defaults.items() if k in accepted} | parse_params(args.param)
        print(
            f"{args.name} {args.dataset} resample={args.resample} "
            f"{cls.__module__}.{cls_name}({params})",
            flush=True,
        )
        clf = cls(**params)
        start = time.time()
        clf.fit(X_train, y_train)
        fit_ms = int((time.time() - start) * 1000)
        start = time.time()
        probs = clf.predict_proba(X_test)
        predict_ms = int((time.time() - start) * 1000)

    preds = np.asarray(clf.classes_)[np.argmax(probs, axis=1)]
    write_classification_results(
        preds,
        probs,
        y_test,
        args.name,
        args.dataset,
        args.output_dir,
        full_path=False,
        first_line_classifier_name=f"{args.name} ({cls_name})",
        split="TEST",
        resample_id=args.resample,
        time_unit="MILLISECONDS",
        first_line_comment=(
            f"Generated by tscbench-run on {datetime.now():%m/%d/%Y, %H:%M:%S}. "
            f"Encoder dictionary: { {str(c): i for i, c in enumerate(labels)} }."
        ),
        parameter_info=str(params).replace("\n", " "),
        accuracy=float((preds == y_test).mean()),
        fit_time=fit_ms,
        predict_time=predict_ms,
        n_classes=len(labels),
    )
    print(
        f"fit {fit_ms / 1000:.1f}s, predict {predict_ms / 1000:.1f}s, "
        f"accuracy {(preds == y_test).mean():.4f}",
        flush=True,
    )


# ----------------------------- driver -----------------------------


def parse_resamples(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        out.extend(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    return sorted(set(out))


def parse_datasets(values):
    """['ucr:Beef,ucr:Coffee', ...] -> [('ucr', 'Beef'), ('ucr', 'Coffee')]; 'ucr' -> all of ucr."""
    out = []
    for value in values:
        for item in filter(None, (x.strip() for x in value.split(","))):
            source, sep, name = item.partition(":")
            if source in DATA_DIRS and not name:
                out.extend(
                    (source, d.name) for d in sorted(DATA_DIRS[source].iterdir()) if d.is_dir()
                )
                continue
            if not sep or source not in DATA_DIRS or not name:
                raise SystemExit(
                    f"dataset {item!r} is not <source>:<name>, source one of {list(DATA_DIRS)}"
                )
            if not (DATA_DIRS[source] / name).is_dir():
                raise SystemExit(f"{item}: no folder {DATA_DIRS[source] / name}")
            out.append((source, name))
    return out


def resolve_repo(repo, ref):
    """Branch, tag or commit of a git repo -> full commit SHA."""
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    out = subprocess.run(
        ["git", "ls-remote", repo, ref, f"{ref}^{{}}"], check=True, capture_output=True, text=True
    ).stdout.split()
    if out:
        return out[::2][-1]  # the peeled commit of an annotated tag comes last
    if re.fullmatch(r"[0-9a-f]{7,39}", ref):
        # a short commit: ls-remote only lists refs, so expand it from a shallow-free fetch
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(
                ["git", "clone", "-q", "--filter=blob:none", "--no-checkout", repo, d], check=True
            )
            return subprocess.run(
                ["git", "-C", d, "rev-parse", f"{ref}^{{commit}}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
    raise SystemExit(f"{repo} has no branch, tag or commit {ref!r}")


def build_env(env_dir, spec, extra_packages):
    """venv with `spec` (and `extra_packages`) and their dependencies, plus tsml-eval's writer."""
    subprocess.run(["uv", "venv", "-q", "--python", PYTHON, str(env_dir)], check=True)
    python = str(env_dir / "bin" / "python")
    subprocess.run(
        ["uv", "pip", "install", "-q", "--python", python, spec, *extra_packages], check=True
    )
    subprocess.run(
        ["uv", "pip", "install", "-q", "--python", python, "--no-deps", TSML_EVAL], check=True
    )
    freeze = subprocess.run(
        ["uv", "pip", "freeze", "--python", python], check=True, capture_output=True, text=True
    ).stdout
    return python, freeze


def s3_exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def gpu_name():
    try:
        return (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or "none"
        )
    except (OSError, subprocess.TimeoutExpired):
        return "none"


def main():
    ap = argparse.ArgumentParser(prog="tscbench-run", description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--repo", help="git repo URL, with --ref")
    what.add_argument("--package", help="pip spec, e.g. aeon==1.6.0")
    ap.add_argument("--ref", help="branch, tag or commit of --repo")
    ap.add_argument(
        "--extra", action="append", default=[], help="package extra, e.g. torch; repeatable"
    )
    ap.add_argument(
        "--with",
        dest="with_packages",
        action="append",
        default=[],
        help="extra package for the venv, e.g. torch; repeatable",
    )
    ap.add_argument(
        "--model", required=True, help="module:Class, e.g. tscglue.models:TSCGlueEnhancedV4"
    )
    ap.add_argument("-p", "--param", action="append", default=[], help="key=value, repeatable")
    ap.add_argument(
        "-d",
        "--datasets",
        action="append",
        required=True,
        help="<source>:<name>, comma-separated or repeated, e.g. ucr:Beef,ucr:Coffee; "
        "<source> alone for all its datasets, e.g. ucr",
    )
    ap.add_argument("-r", "--resamples", default="0-9", help="e.g. 0-9 or 0,3,5 (default 0-9)")
    ap.add_argument("--name", help="results name (default: Class-<commit|version>-<params hash>)")
    ap.add_argument(
        "-o",
        "--output-dir",
        default=str(PROJECT_ROOT / "tscbench_runs"),
        help="local dir or s3://bucket/prefix (default: TSCbench/tscbench_runs)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="redo runs whose results exist (default: skip them)",
    )
    ap.add_argument("--keep-env", action="store_true", help="do not delete the venv at the end")
    args = ap.parse_args()

    if args.repo:
        if not args.ref:
            ap.error("--repo needs --ref")
        sha = resolve_repo(args.repo, args.ref)
        dist = args.repo.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1].lower()
        extras = f"[{','.join(args.extra)}]" if args.extra else ""
        spec, label = f"{dist}{extras} @ git+{args.repo}@{sha}", sha[:7]
    else:
        if args.ref:
            ap.error("--ref goes with --repo")
        spec = args.package
        if args.extra:  # aeon==1.6.0 -> aeon[all_extras]==1.6.0
            m = re.match(r"([A-Za-z0-9_.-]+)(.*)", spec)
            spec = f"{m[1]}[{','.join(args.extra)}]{m[2]}"
        label = re.sub(r"[^A-Za-z0-9.]+", "", spec.split("=")[-1]) or "pkg"

    cls_name = args.model.rpartition(":")[2]
    params_hash = hashlib.sha256(" ".join(sorted(args.param)).encode()).hexdigest()[:6]
    name = args.name or f"{cls_name}-{label}-{params_hash}"
    datasets = parse_datasets(args.datasets)
    resamples = parse_resamples(args.resamples)
    s3 = None
    if args.output_dir.startswith("s3://"):
        import boto3

        # fresh session per call re-reads credentials, so refreshed ones (ada) are picked up
        def s3():
            return boto3.Session().client("s3")

        bucket, _, prefix = args.output_dir.removeprefix("s3://").partition("/")
        key_root = f"{prefix.strip('/')}/{name}".lstrip("/")
        pages = s3().get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=f"{key_root}/Predictions/"
        )
        uploaded = {
            o["Key"].removeprefix(f"{key_root}/") for p in pages for o in p.get("Contents", [])
        }
        results_url = f"s3://{bucket}/{key_root}"
    else:
        results_url = Path(args.output_dir) / name

    def is_done(dataset, resample, fresh=False):
        rel = f"Predictions/{dataset}/testResample{resample}.csv"
        if s3 is None:
            return (results_url / rel).exists()
        return s3_exists(s3(), bucket, f"{key_root}/{rel}") if fresh else rel in uploaded

    runs = [
        (s, d, r) for s, d in datasets for r in resamples if args.overwrite or not is_done(d, r)
    ]
    random.shuffle(runs)
    print(f"{name}: {spec}  {args.model}  {args.param}")
    print(
        f"{len(runs)} of {len(datasets) * len(resamples)} runs to do, results in {results_url}",
        flush=True,
    )
    if not runs:
        return

    if s3 is None:
        results, lock_dir = results_url, results_url / "locks"
    else:  # stage locally, upload each run as it ends; locks shared by copies on this machine
        results = Path(tempfile.mkdtemp(prefix="tscbench-run-out-")) / name
        lock_dir = Path(tempfile.gettempdir()) / "tscbench-run-locks" / name

    def upload(rel):
        if s3 is not None and (results / rel).exists():
            s3().upload_file(str(results / rel), bucket, f"{key_root}/{rel}")

    env_dir = Path(tempfile.mkdtemp(prefix="tscbench-run-env-"))
    try:
        print(f"building venv in {env_dir}", flush=True)
        python, freeze = build_env(env_dir, spec, args.with_packages)
        (results / "logs").mkdir(parents=True, exist_ok=True)
        lock_dir.mkdir(parents=True, exist_ok=True)
        (results / "run_info.txt").write_text(
            f"date: {datetime.now():%Y-%m-%d %H:%M:%S}\nhost: {socket.gethostname()}\n"
            f"gpu: {gpu_name()}\npython: {platform.python_version()} (venv {PYTHON})\n"
            f"package: {spec}\nwith: {args.with_packages}\n"
            + (f"repo: {args.repo}\nref: {args.ref}\ncommit: {sha}\n" if args.repo else "")
            + f"model: {args.model}\nparams: {args.param}\n"
            f"datasets: {','.join(f'{s}:{d}' for s, d in datasets)}\nresamples: {args.resamples}\n"
            f"\n# venv packages\n{freeze}"
        )
        upload("run_info.txt")
        run_env = os.environ | {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }

        ok, skipped = [], 0
        for source, dataset, resample in runs:
            # another copy may be running it (lock held) or have finished it since we started
            lock = open(lock_dir / f"{dataset}_{resample}.lock", "w")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                skipped += 1
                continue
            if not args.overwrite and is_done(dataset, resample, fresh=True):
                lock.close()
                skipped += 1
                continue
            cmd = [
                python,
                str(Path(__file__).resolve()),
                "_run",
                "--model",
                args.model,
                "--name",
                name,
                "--dataset",
                dataset,
                "--resample",
                str(resample),
                "--output-dir",
                str(results.parent),
                "--data-dir",
                str(DATA_DIRS[source]),
                *[x for p in args.param for x in ("-p", p)],
            ]
            with open(results / "logs" / f"{dataset}_{resample}.log", "w") as log:
                code = subprocess.run(
                    cmd, env=run_env, stdout=log, stderr=subprocess.STDOUT
                ).returncode
            upload(f"Predictions/{dataset}/testResample{resample}.csv")
            upload(f"logs/{dataset}_{resample}.log")
            lock.close()
            print(
                f"{'ok  ' if code == 0 else 'FAIL'} {source}:{dataset} resample={resample}",
                flush=True,
            )
            ok.append(code == 0)
        print(
            f"done: {sum(ok)} ok, {len(ok) - sum(ok)} failed, {skipped} done or running elsewhere "
            f"(logs in {results_url}/logs)"
        )
        sys.exit(0 if all(ok) else 1)
    finally:
        if s3 is not None:
            shutil.rmtree(results.parent, ignore_errors=True)
        if args.keep_env:
            print(f"kept venv {env_dir}")
        else:
            shutil.rmtree(env_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_run":
        rp = argparse.ArgumentParser()
        for flag in ("--model", "--name", "--dataset", "--output-dir", "--data-dir"):
            rp.add_argument(flag, required=True)
        rp.add_argument("--resample", type=int, required=True)
        rp.add_argument("-p", "--param", action="append", default=[])
        run_one(rp.parse_args(sys.argv[2:]))
    else:
        main()
