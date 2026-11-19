"""Markdown report of the fold-method benchmark (run_fold_method.py): stratified k-fold vs round-robin.

    # print the report: 50 smallest UCR datasets, folds 0-9
    .venv-fold-method/bin/python scripts/report_fold_method.py --smallest 50 -f 0-9
    # also post it as a comment on TSCGlue PR #20
    .venv-fold-method/bin/python scripts/report_fold_method.py --smallest 50 -f 0-9 --post

Only (dataset, fold) pairs where both fold methods have results for a head are compared.
Δ = round-robin - stratified k-fold.
"""

import subprocess
import sys
from collections import Counter
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_fold_method import (  # noqa: E402
    DATA_DIR, FOLD_METHODS, METRICS, OUTPUT_DIR, TSCGLUE_COMMIT, n_train, parse_folds, result_path,
)

SKF, RR = "stratified-kfold", "round-robin"
K_FOLDS = 10


def class_counts(dataset, data_dir):
    with open(Path(data_dir) / dataset / f"{dataset}0_TRAIN.ts") as f:
        for line in f:
            if line.strip().lower() == "@data":
                break
        return Counter(line.rsplit(":", 1)[1].strip() for line in f if line.strip())


def scores(path):
    """Accuracy, log loss, macro F1 and AUROC of one tsml_eval results file."""
    from tsml_eval.evaluation.storage import load_classifier_results

    res = load_classifier_results(str(path))
    return res.accuracy, res.log_loss, res.f1_score, res.auroc_score


def wilcoxon_p(a, b):
    from scipy.stats import wilcoxon

    d = np.asarray(a) - np.asarray(b)
    if len(d) == 0 or np.all(np.abs(d) < 1e-12):
        return float("nan")
    return float(wilcoxon(a, b, zero_method="zsplit").pvalue)


def summary_row(label, skf, rr, higher_better):
    skf, rr = np.asarray(skf), np.asarray(rr)
    d = rr - skf
    better = int(np.sum(d > 1e-9) if higher_better else np.sum(d < -1e-9))
    tie = int(np.sum(np.abs(d) < 1e-9))
    worse = len(d) - better - tie
    p = wilcoxon_p(rr, skf)
    return (f"| {label} | {len(d)} | {skf.mean():.4f} | {rr.mean():.4f} | {d.mean():+.4f} | "
            f"{better} / {tie} / {worse} | {'n/a' if np.isnan(p) else f'{p:.3g}'} |")


def fmt(x, signed=False):
    return "–" if x is None or np.isnan(x) else (f"{x:+.4f}" if signed else f"{x:.4f}")


def wdl(d, higher_better):
    """Round-robin win / draw / loss counts for the differences d = RR - SKF."""
    d = np.asarray(d)
    win = int(np.sum(d > 1e-9) if higher_better else np.sum(d < -1e-9))
    draw = int(np.sum(np.abs(d) < 1e-9))
    return win, draw, len(d) - win - draw


PR_MARKER = "<!-- fold-method-results -->"
TABLES = [  # (head, metric index in scores(), title, higher is better)
    ("Accuracy", 0, "High-Accuracy head · accuracy ↑", True),
    ("Accuracy", 2, "High-Accuracy head · F1 (macro) ↑", True),
    ("Accuracy", 3, "High-Accuracy head · AUROC ↑", True),
    ("LogLoss", 1, "High-LogLoss head · log loss ↓", False),
]


def pr_body_section(names, found, fold_ids, data_dir, done, total):
    """Four per-dataset tables (head x metric) with round-robin win / draw / loss vs stratified k-fold."""
    from datetime import datetime

    lines = [
        PR_MARKER,
        "## Benchmark: stratified k-fold vs round-robin",
        "",
        f"Preset `high`, {len(names)} smallest UCR datasets, resamples {fold_ids[0]}-{fold_ids[-1]}, "
        f"mean over the runs (resamples) finished for both, Δ = round-robin − stratified-kfold. {done} / {total} jobs done, updated {datetime.now():%Y-%m-%d %H:%M}.",
    ]
    for head, i, title, higher_better in TABLES:
        lines += ["", f"### {title}", "", "| dataset | runs | stratified-kfold | round-robin | Δ |",
                  "|---|---|---|---|---|"]
        skfs, rrs, runs = [], [], 0
        for d in sorted(names, key=str.lower):
            common = sorted(set(found[(head, SKF)][d]) & set(found[(head, RR)][d]))
            if not common:
                lines.append(f"| {d} | 0 | – | – | – |")
                continue
            s = np.mean([found[(head, SKF)][d][f][i] for f in common])
            r = np.mean([found[(head, RR)][d][f][i] for f in common])
            skfs.append(s)
            rrs.append(r)
            runs += len(common)
            lines.append(f"| {d} | {len(common)} | {s:.4f} | {r:.4f} | {r - s:+.4f} |")
        diffs = np.array(rrs) - np.array(skfs)
        if skfs:
            lines.append(f"| **mean** | {runs} | **{np.mean(skfs):.4f}** | **{np.mean(rrs):.4f}** | "
                         f"**{diffs.mean():+.4f}** |")
        w, dr, lo = wdl(diffs, higher_better)
        lines += ["", f"{runs} / {len(names) * len(fold_ids)} runs compared. "
                  f"round-robin win / draw / loss: **{w} / {dr} / {lo}**"]
    return "\n".join(lines) + "\n"


def update_pr_body(section, output_dir):
    """Replace everything after PR_MARKER in PR #20's description with section (append if no marker yet)."""
    repo = ["--repo", "gasperpetelin/TSCGlue"]
    body = subprocess.run(["gh", "pr", "view", "20", *repo, "--json", "body", "-q", ".body"],
                          check=True, capture_output=True, text=True).stdout.rstrip("\n")
    head = body.split(PR_MARKER)[0].rstrip("\n")
    path = Path(output_dir) / "pr_body.md"
    path.write_text(f"{head}\n\n{section}")
    subprocess.run(["gh", "pr", "edit", "20", *repo, "--body-file", str(path)], check=True,
                   stdout=subprocess.DEVNULL)


@click.command()
@click.option("-o", "--output-dir", default=str(OUTPUT_DIR), show_default=True)
@click.option("-d", "--datasets", default=None, help="Comma-separated datasets. Default: all 112 UCR equal-length datasets.")
@click.option("--smallest", type=int, default=None, help="Keep only the N datasets with the fewest training cases.")
@click.option("-f", "--folds", default="0-29", show_default=True, help="Folds, e.g. 0-9 or 0,3,5.")
@click.option("--data-dir", default=str(DATA_DIR), show_default=True)
@click.option("--post", is_flag=True, help="Post the report as a comment on gasperpetelin/TSCGlue PR #20.")
@click.option("--pr-body", is_flag=True, help="Write the four per-dataset tables into PR #20's description instead.")
def main(output_dir, datasets, smallest, folds, data_dir, post, pr_body):
    from aeon.datasets.tsc_datasets import univariate_equal_length

    names = [d.strip() for d in datasets.split(",") if d.strip()] if datasets else sorted(univariate_equal_length)
    if smallest:
        names = sorted(names, key=lambda d: (n_train(d, data_dir), d))[:smallest]
    names = sorted(names, key=lambda d: (n_train(d, data_dir), d))
    fold_ids = parse_folds(folds)

    # scores[(metric, method)][dataset][fold] = (accuracy, log loss)
    found = {(m, fm): {d: {} for d in names} for m in METRICS for fm in FOLD_METHODS}
    missing_jobs = 0
    for d in names:
        for fold in fold_ids:
            for fm in FOLD_METHODS:
                paths = {m: result_path(output_dir, m, fm, d, fold) for m in METRICS}
                missing_jobs += not all(p.exists() for p in paths.values())
                for m, p in paths.items():
                    if p.exists():
                        found[(m, fm)][d][fold] = scores(p)

    if pr_body:
        total = len(names) * len(fold_ids) * len(FOLD_METHODS)
        section = pr_body_section(names, found, fold_ids, data_dir, total - missing_jobs, total)
        click.echo(section)
        update_pr_body(section, output_dir)
        return

    # per-dataset means over folds where both fold methods have results for that head
    means = {}  # (metric, dataset) -> (n folds, acc skf, acc rr, logloss skf, logloss rr)
    for m in METRICS:
        for d in names:
            common = sorted(set(found[(m, SKF)][d]) & set(found[(m, RR)][d]))
            if common:
                s = np.mean([found[(m, SKF)][d][f] for f in common], axis=0)
                r = np.mean([found[(m, RR)][d][f] for f in common], axis=0)
                means[(m, d)] = (len(common), s[0], r[0], s[1], r[1])
    pairs = {m: sum(v[0] for (mm, _), v in means.items() if mm == m) for m in METRICS}
    n_data = {m: sum(1 for (mm, _) in means if mm == m) for m in METRICS}

    def col(m, i):
        return [means[(m, d)][i] for d in names if (m, d) in means]

    which = f"{smallest} smallest UCR datasets by train size" if smallest else f"{len(names)} UCR datasets"
    lines = [
        "## Fold-method benchmark: stratified k-fold vs round-robin",
        "",
        f"`TSCGlueEnhancedV4`, preset `high`, at this PR's commit `{TSCGLUE_COMMIT}`; {which}, "
        f"resamples {folds} (predefined UCR resamples), {K_FOLDS}-fold internal CV, both heads "
        "(High-Accuracy: `eval_metric=accuracy`, High-LogLoss: `eval_metric=log_loss`), GPU, `n_jobs=8`. "
        "SKF = `fold_method=\"stratified-kfold\"` (default), RR = `fold_method=\"round-robin\"`. "
        "Δ = RR − SKF.",
        "",
        f"Completed (dataset, resample) pairs with both fold methods: High-Accuracy {pairs['Accuracy']} "
        f"over {n_data['Accuracy']} datasets, High-LogLoss {pairs['LogLoss']} over {n_data['LogLoss']} datasets "
        f"(of {len(names)} × {len(fold_ids)} = {len(names) * len(fold_ids)}). "
        f"Failed or missing jobs: {missing_jobs} of {len(names) * len(fold_ids) * len(FOLD_METHODS)}.",
        "",
        "### Summary (per-dataset means over resamples)",
        "",
        "Better / tie / worse counts datasets where RR is better / equal (|Δ| < 1e-9) / worse; "
        "p is a Wilcoxon signed-rank test (zsplit) over per-dataset means.",
        "",
        "| Head · metric | datasets | SKF | RR | mean Δ | RR better / tie / worse | Wilcoxon p |",
        "|---|---|---|---|---|---|---|",
    ]
    if n_data["Accuracy"]:
        lines.append(summary_row("**High-Accuracy · accuracy** ↑", col("Accuracy", 1), col("Accuracy", 2), True))
    if n_data["LogLoss"]:
        lines.append(summary_row("**High-LogLoss · log loss** ↓", col("LogLoss", 3), col("LogLoss", 4), False))
    if n_data["LogLoss"]:
        lines.append(summary_row("High-LogLoss · accuracy ↑", col("LogLoss", 1), col("LogLoss", 2), True))
    if n_data["Accuracy"]:
        lines.append(summary_row("High-Accuracy · log loss ↓", col("Accuracy", 3), col("Accuracy", 4), False))

    lines += [
        "",
        "### Per dataset",
        "",
        "Accuracy from the High-Accuracy head, log loss from the High-LogLoss head; means over the "
        "resamples completed for both fold methods. Class counts from the resample-0 train split. "
        f"Flags: **F** = every class has < {K_FOLDS} cases, so SKF falls back to unstratified KFold; "
        "**1** = a class has a single case (tscglue then skips internal CV and fits its MRHydraET "
        "fallback, so fold_method has no effect); **s** = smallest class < "
        f"{K_FOLDS} (StratifiedKFold runs but some validation folds miss that class).",
        "",
        "| dataset | flags | n_train | classes | min class | resamples | acc SKF | acc RR | Δacc | "
        "logloss SKF | logloss RR | Δlogloss |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in names:
        counts = class_counts(d, data_dir)
        flags = ("F" if max(counts.values()) < K_FOLDS else "") + ("1" if min(counts.values()) == 1 else "")
        if not flags and min(counts.values()) < K_FOLDS:
            flags = "s"
        a = means.get(("Accuracy", d), (0, np.nan, np.nan, np.nan, np.nan))
        ll = means.get(("LogLoss", d), (0, np.nan, np.nan, np.nan, np.nan))
        res = str(a[0]) if a[0] == ll[0] else f"{a[0]} / {ll[0]}"
        lines.append(
            f"| {d} | {flags} | {n_train(d, data_dir)} | {len(counts)} | {min(counts.values())} | {res} | "
            f"{fmt(a[1])} | {fmt(a[2])} | {fmt(a[2] - a[1], True)} | "
            f"{fmt(ll[3])} | {fmt(ll[4])} | {fmt(ll[4] - ll[3], True)} |"
        )
    lines += ["", "🤖 Generated with [Claude Code](https://claude.com/claude-code)", ""]
    report = "\n".join(lines)
    click.echo(report)

    if post:
        body = Path(output_dir) / "report.md"
        body.write_text(report)
        subprocess.run(["gh", "pr", "comment", "20", "--repo", "gasperpetelin/TSCGlue", "--body-file", str(body)],
                       check=True)


# tsml_eval / tscglue imports may start worker processes: keep the run behind the guard.
if __name__ == "__main__":
    main()
