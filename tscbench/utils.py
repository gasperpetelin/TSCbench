"""Cache utilities for benchmark result files."""

from __future__ import annotations

from aeon.datasets import load_classification
import os
import tempfile
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import polars as pl

class MemoryTracker:
    """Track peak RSS memory (process + children) in a background thread."""

    def __init__(self, interval=0.5):
        self.peak = 0
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        import time
        import psutil
        proc = psutil.Process(os.getpid())
        self._times = []
        self._values = []
        t0 = None
        while not self._stop.wait(self._interval):
            now = time.monotonic()
            if t0 is None:
                t0 = now
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            self.peak = max(self.peak, total)
            self._times.append(now - t0)
            self._values.append(total)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        return self.peak

    def plot(self, path="memory.png"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self._times, [v / 1024**3 for v in self._values])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Memory (GB)")
        ax.set_title(f"Peak: {self.peak / 1024**3:.2f} GB")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        print(f"Memory plot saved to {path}")
        plt.close(fig)


class AbstractFileCache(ABC):
    """Common interface for local and remote parquet result caches."""

    @abstractmethod
    def exists(self, filename: str) -> bool:
        """Return whether a cached file exists."""

    @abstractmethod
    def add(self, df: pl.DataFrame, filename: str) -> None:
        """Write a dataframe to the cache."""

    @abstractmethod
    def list_files(self) -> list[str]:
        """Return cached parquet filenames."""

    @abstractmethod
    def read_parquet(self, filename: str) -> pl.DataFrame:
        """Read one parquet file from the cache."""

    def read_all_parquet(self) -> pl.DataFrame:
        files = sorted(name for name in self.list_files() if name.endswith(".parquet"))
        if not files:
            return pl.DataFrame()
        return pl.concat([self.read_parquet(name) for name in files], how="diagonal_relaxed")


class S3FileCache(AbstractFileCache):
    def __init__(self, base_s3_dir: str):
        if not base_s3_dir.startswith("s3://"):
            raise ValueError("S3FileCache requires an s3:// URI")

        import boto3

        self.base_s3_dir = base_s3_dir.rstrip("/")
        parsed = urlparse(base_s3_dir)
        self.bucket = parsed.netloc
        self.prefix = parsed.path.lstrip("/").rstrip("/")
        self._s3 = boto3.client("s3")
        self._files: set[str] = set()
        self._loaded = False

    def _full_key(self, filename: str) -> str:
        return f"{self.prefix}/{filename}" if self.prefix else filename

    def _load_once(self) -> None:
        if self._loaded:
            return

        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key[len(self.prefix) + 1 :] if self.prefix else key
                self._files.add(filename)
        self._loaded = True

    def exists(self, filename: str) -> bool:
        from botocore.exceptions import ClientError

        self._load_once()
        if filename in self._files:
            return True

        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._full_key(filename))
            self._files.add(filename)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            raise

    def add(self, df: pl.DataFrame, filename: str) -> None:
        df.write_parquet(f"{self.base_s3_dir}/{filename}")
        self._files.add(filename)

    def list_files(self) -> list[str]:
        self._load_once()
        return sorted(self._files)

    def read_parquet(self, filename: str) -> pl.DataFrame:
        return pl.read_parquet(f"{self.base_s3_dir}/{filename}")

    def read_all_parquet_cached(
        self, cache_dir: str | Path | None = None, max_workers: int = 16
    ) -> pl.DataFrame:
        from tqdm import tqdm

        subdir = f"{self.bucket}__{self.prefix.replace('/', '_')}"
        local_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(tempfile.gettempdir()) / "tscbench-cache" / subdir
        )
        local_dir.mkdir(parents=True, exist_ok=True)

        local_files = {path.name for path in local_dir.iterdir() if path.suffix == ".parquet"}
        self._load_once()
        remote_new = [
            filename
            for filename in self._files
            if filename.endswith(".parquet") and filename not in local_files
        ]

        def download(filename: str) -> None:
            self._s3.download_file(self.bucket, self._full_key(filename), str(local_dir / filename))

        if remote_new:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(download, filename) for filename in remote_new]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    future.result()

        paths = sorted(local_dir.glob("*.parquet"))
        if not paths:
            return pl.DataFrame()
        return pl.read_parquet(paths)


class LogsFileCache(AbstractFileCache):
    """No-op cache that just prints each operation — useful for dry runs."""

    def exists(self, filename: str) -> bool:
        print(f"[LogsFileCache] exists? {filename} → False")
        return False

    def add(self, df: pl.DataFrame, filename: str) -> None:
        with pl.Config(tbl_rows=-1, tbl_cols=-1):
            print(f"[LogsFileCache] add {filename}\n{df}")

    def list_files(self) -> list[str]:
        print("[LogsFileCache] list_files → []")
        return []

    def read_parquet(self, filename: str) -> pl.DataFrame:
        print(f"[LogsFileCache] read_parquet {filename} → empty DataFrame")
        return pl.DataFrame()


class LocalFileCache(AbstractFileCache):
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def exists(self, filename: str) -> bool:
        return (self.base_dir / filename).exists()

    def add(self, df: pl.DataFrame, filename: str) -> None:
        target = self.base_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            df.write_parquet(tmp_path)
            tmp_path.replace(target)
        finally:
            tmp_path.unlink(missing_ok=True)

    def list_files(self) -> list[str]:
        return sorted(path.name for path in self.base_dir.glob("*.parquet"))

    def read_parquet(self, filename: str) -> pl.DataFrame:
        return pl.read_parquet(self.base_dir / filename)

def software_versions() -> dict:
    import sys
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    def _ver(name: str) -> str | None:
        try:
            return pkg_version(name)
        except PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "tscbench": _ver("tscbench"),
        "tscglue": _ver("tscglue"),
        "aeon": _ver("aeon"),
        "sklearn": _ver("scikit-learn"),
        "numpy": _ver("numpy"),
        "polars": _ver("polars"),
    }


def hardware_info() -> dict:
    cpu_name = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    gpu_names = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass

    import psutil
    return {
        "cpu": cpu_name,
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "gpu": str(gpu_names) if gpu_names else None,
    }


def discover_datasets(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        return []
    return sorted(path.name for path in data_dir.iterdir() if path.is_dir())


def load_dataset(dataset_name):
    """Load and normalize a dataset."""
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")
    return X_train, y_train, X_test, y_test

def load_s3_parquet_cached(
    s3_prefix: str = "s3://tsc-glue/performance-benchmarking/",
    max_workers: int = 16,
    skip_empty: bool = False,
) -> pl.DataFrame:
    """Download all parquet files from an S3 prefix, cache locally, and return as a DataFrame."""
    import boto3
    from tqdm import tqdm

    cache_dir = os.path.join(tempfile.gettempdir(), "tscglue-cache")
    os.makedirs(cache_dir, exist_ok=True)

    parsed = urlparse(s3_prefix)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    s3 = boto3.client("s3")

    local_files = {f for f in os.listdir(cache_dir) if f.endswith(".parquet")}

    paginator = s3.get_paginator("list_objects_v2")
    remote_keys = []
    skipped = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                if skip_empty and obj["Size"] == 0:
                    skipped += 1
                    continue
                fname = key.rsplit("/", 1)[-1]
                if fname not in local_files:
                    remote_keys.append(key)

    if skip_empty and skipped:
        print(f"Skipped {skipped} empty parquet file(s) on S3")

    def _download(key):
        fname = key.rsplit("/", 1)[-1]
        s3.download_file(bucket, key, os.path.join(cache_dir, fname))

    if remote_keys:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_download, k) for k in remote_keys]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass

    local_paths = sorted(
        os.path.join(cache_dir, f)
        for f in os.listdir(cache_dir)
        if f.endswith(".parquet") and (not skip_empty or os.path.getsize(os.path.join(cache_dir, f)) > 0)
    )
    return pl.read_parquet(local_paths)

def load_ucr_fold(data_dir: Path, dataset_name: str, fold: int):
    from aeon.datasets import load_from_ts_file
    #if dataset_name.startswith("m-"):
    #    return load_tscglue_fold(dataset_name, fold)
    train = data_dir / dataset_name / f"{dataset_name}{fold}_TRAIN.ts"
    test  = data_dir / dataset_name / f"{dataset_name}{fold}_TEST.ts"
    if train.exists() and test.exists():
        X_train, y_train = load_from_ts_file(train)
        X_test,  y_test  = load_from_ts_file(test)
        return X_train, y_train, X_test, y_test
    else:
        raise FileNotFoundError(f"Fold files not found for dataset={dataset_name} fold={fold} in {data_dir}")
    #return load_tscglue_fold(dataset_name, fold)