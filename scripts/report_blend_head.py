"""Markdown report of run_blend_head.py: PR #23's best preset (blend) vs current main, and PR update.

    # print the report: 50 smallest UCR datasets, folds 0-9
    .venv/bin/python scripts/report_blend_head.py --smallest 50 -f 0-9
    # also replace the "Benchmark" section of TSCGlue PR #23's description with it
    .venv/bin/python scripts/report_blend_head.py --smallest 50 -f 0-9 --update-pr

Only (dataset, fold) pairs where both compared classifiers have results are compared.
Scores come from the saved test probabilities. W/D/L counts datasets (mean over their folds).
"""

import re
import subprocess
import sys
from pathlib import Path

import click
import numpy as np
from sklearn.metrics import f1_score, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_blend_head import DATA_DIR, OUTPUT_DIR, VERSIONS, parse_folds, result_path, smallest_datasets  # noqa: E402

NEW = ("Best-LogLoss", "new")
OLD = [("High-Accuracy", "old"), ("High-LogLoss", "old"), ("Best-LogLoss", "old")]
LABELS = {
    NEW: "PR Best (blend)",
    ("High-Accuracy", "old"): "main High-Accuracy",
    ("High-LogLoss", "old"): "main High-LogLoss",
    ("Best-LogLoss", "old"): "main Best (meta-probability-et)",
}
METRICS = [("accuracy", 1), ("f1", 1), ("auroc", 1), ("log_loss", -1)]
SECTION = "## Benchmark: before vs after"
REPO, PR = "gasperpetelin/TSCGlue", 23


def scores(path):
    """Accuracy, macro F1, AUROC and log loss from a tsml results file's test probabilities."""
    lines = Path(path).read_text().splitlines()
    k = int(lines[2].split(",")[5])
    rows = [x.split(",") for x in lines[3:] if x.strip()]
    y = np.array([int(x[0]) for x in rows])
    p = np.clip(np.array([[float(v) for v in x[3:3 + k]] for x in rows]), 1e-12, 1)
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    auc = roc_auc_score(y, p[:, 1]) if k == 2 else roc_auc_score(y, p, multi_class="ovr", labels=np.arange(k))
    return {"accuracy": float((pred == y).mean()), "f1": f1_score(y, pred, average="macro"),
            "auroc": auc, "log_loss": log_loss(y, p, labels=np.arange(k))}


def report(datasets, folds, output_dir):
    cache = {}

    def get(clf, ds, f):
        key = (clf, ds, f)
        if key not in cache:
            path = result_path(output_dir, clf[0], clf[1], ds, f)
            cache[key] = scores(path) if path.exists() else None
        return cache[key]

    n_jobs = len(datasets) * len(folds) * len(VERSIONS)
    done = sum(all(result_path(output_dir, b, v, ds, f).exists() for b in spec["classifiers"])
               for ds in datasets for f in folds for v, spec in VERSIONS.items())
    which = ("all 112 UCR equal-length datasets" if len(datasets) == 112
             else f"{len(datasets)} smallest UCR datasets by train size")
    out = [SECTION, "",
           f"`TSCGlueEnhancedV4` on jastreb (GPU, `n_jobs=8`), {which}, "
           f"resamples {folds[0]}-{folds[-1]}, 10-fold internal CV. main = `80a54c0`, "
           f"PR = `bc444fb`. PR Best serves the same blend for every `eval_metric`, so it is fit once. "
           f"Datasets where TSCGlue falls back to MRHydraET (a class with < 2 cases) give identical "
           f"results in both versions.", "",
           f"Progress: {done} of {n_jobs} jobs done.", ""]
    head = "| PR Best vs | Datasets | " + " | ".join(
        f"{m} (main → PR, W/D/L)" for m in ["Accuracy", "F1", "AUROC", "Log loss"]) + " |"
    out += [head, "|---|---|" + "---|" * 4]
    for old in OLD:
        per = {}
        for ds in datasets:
            pairs = [(get(old, ds, f), get(NEW, ds, f)) for f in folds]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if pairs:
                per[ds] = {m: (np.mean([a[m] for a, _ in pairs]), np.mean([b[m] for _, b in pairs]))
                           for m, _ in METRICS}
        if not per:
            out.append(f"| {LABELS[old]} | 0 | " + " | ".join("–" for _ in METRICS) + " |")
            continue
        cells = []
        for m, sign in METRICS:
            a = np.array([v[m][0] for v in per.values()])
            b = np.array([v[m][1] for v in per.values()])
            d = sign * (b - a)
            w, l = int((d > 1e-9).sum()), int((d < -1e-9).sum())
            cells.append(f"{a.mean():.4f} → {b.mean():.4f}, {w}/{len(d) - w - l}/{l}")
        out.append(f"| {LABELS[old]} | {len(per)} | " + " | ".join(cells) + " |")
    out += ["", "W/D/L = datasets where PR Best is better / equal / worse (lower log loss is better)."]

    # average rank of all four, on (dataset, fold) pairs where every one of them has results
    clfs = OLD + [NEW]
    per = {}
    for ds in datasets:
        runs = [[get(c, ds, f) for c in clfs] for f in folds]
        runs = [r for r in runs if all(s is not None for s in r)]
        if runs:
            per[ds] = {m: [np.mean([r[i][m] for r in runs]) for i in range(len(clfs))] for m, _ in METRICS}
    out += ["", f"Average rank among the four (1 = best; ties share the mean rank), {len(per)} datasets:", "",
            "| | Accuracy | F1 | AUROC | Log loss |", "|---|---|---|---|---|"]
    if per:
        ranks = {}
        for m, sign in METRICS:
            vals = np.array([per[ds][m] for ds in per]) * -sign  # lower = better for ranking
            order = np.argsort(np.argsort(vals, axis=1), axis=1).astype(float) + 1
            for row, v in zip(order, vals):  # average the ranks of ties
                for x in np.unique(v):
                    row[v == x] = row[v == x].mean()
            ranks[m] = order.mean(axis=0)
        for i, c in enumerate(clfs):
            out.append(f"| {LABELS[c]} | " + " | ".join(f"{ranks[m][i]:.2f}" for m, _ in METRICS) + " |")
    return "\n".join(out)


def update_pr(section):
    body = subprocess.run(["gh", "pr", "view", str(PR), "-R", REPO, "--json", "body", "-q", ".body"],
                          check=True, capture_output=True, text=True).stdout
    sig = "🤖 Generated with"
    main, _, tail = body.partition(sig)
    main = re.split(re.escape(SECTION), main)[0].rstrip()
    new_body = f"{main}\n\n{section}\n\n{sig}{tail}" if tail else f"{main}\n\n{section}\n"
    subprocess.run(["gh", "pr", "edit", str(PR), "-R", REPO, "--body-file", "-"],
                   input=new_body, text=True, check=True, capture_output=True)


@click.command()
@click.option("--smallest", type=int, default=50, show_default=True)
@click.option("-f", "--folds", default="0-9", show_default=True)
@click.option("-o", "--output-dir", default=str(OUTPUT_DIR), show_default=True)
@click.option("--update-pr", "push", is_flag=True,
              help=f"Replace the benchmark section of {REPO}#{PR}'s description.")
def cli(smallest, folds, output_dir, push):
    section = report(smallest_datasets(smallest, DATA_DIR), parse_folds(folds), output_dir)
    click.echo(section)
    if push:
        update_pr(section)
        click.echo(f"\nUpdated {REPO}#{PR}")


if __name__ == "__main__":
    cli()
