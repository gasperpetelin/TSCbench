"""Quick smoke test: run TSCGlue on one dataset fold and print GPU info.

Example:
    uv run python scripts/test_gpu.py
    uv run python scripts/test_gpu.py --dataset ArrowHead --fold 0
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click
import torch
from tscglue.data_loader import DATA_DIR as TSCGLUE_DATA_DIR
from tscglue.models import TSCGlue
from tscbench.utils import load_ucr_fold


@click.command()
@click.option("--dataset", default="ArrowHead", show_default=True)
@click.option("--fold", default=0, show_default=True, type=int)
@click.option("--data-dir", default="data", show_default=True, type=click.Path(path_type=Path))
def main(dataset, fold, data_dir):
    click.echo(f"PyTorch:  {torch.__version__}")
    click.echo(f"CUDA:     {torch.cuda.is_available()} ({torch.cuda.device_count()} device(s))")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            click.echo(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    local_data_dir = data_dir if Path(data_dir).exists() else Path(TSCGLUE_DATA_DIR)
    click.echo(f"\nDataset:  {dataset}  fold={fold}  data_dir={local_data_dir}")

    X_train, y_train, X_test, y_test = load_ucr_fold(local_data_dir, dataset, fold)
    click.echo(f"Train:    {X_train.shape}  Test: {X_test.shape}")

    model = TSCGlue(random_state=fold)
    click.echo("Fitting...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = (preds == y_test).mean()
    click.echo(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
