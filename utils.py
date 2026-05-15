"""Compatibility wrapper for cache utilities.

New code should import from ``tscbench.utils``.
"""

from tscbench.utils import AbstractFileCache, LocalFileCache, S3FileCache

__all__ = ["AbstractFileCache", "LocalFileCache", "S3FileCache"]
