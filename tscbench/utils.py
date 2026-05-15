"""Cache utilities for benchmark result files."""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import polars as pl


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
        return pl.concat([self.read_parquet(name) for name in files], how="diagonal")


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

    def read_all_parquet(self) -> pl.DataFrame:
        return pl.read_parquet(f"{self.base_s3_dir}/*.parquet")

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
