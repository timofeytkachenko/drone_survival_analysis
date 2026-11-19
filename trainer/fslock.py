"""Advisory file locks shared by the studio, the CLI and the API (POSIX only)."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class LockBusyError(RuntimeError):
    """Raised when a non-blocking lock is held by another process or thread."""


@contextmanager
def locked(path: Path, blocking: bool = True) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``path`` for the duration of the block.

    The lock is tied to the open file, so it is released when the process dies.

    Parameters
    ----------
    path : Path
        Lock file; created with its parent directory if missing.
    blocking : bool, default True
        Wait for the lock; with ``False`` raise ``LockBusyError`` if it is held.

    Raises
    ------
    LockBusyError
        If ``blocking`` is false and the lock is held elsewhere.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise LockBusyError(f"{path.name} is held by another process") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
