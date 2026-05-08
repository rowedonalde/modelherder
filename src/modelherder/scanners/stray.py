from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator

from modelherder.models import (
    STRAY_EXTS,
    ModelEntry,
    SOURCE_OTHER,
    format_for_path,
)


# Directories whose names we never descend into, regardless of where they
# show up. Keeps the recursive scan from chewing through dependency caches,
# build outputs, and OS internals.
NOISE_DIR_NAMES = frozenset({
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "Library",         # macOS user library — large and indexed elsewhere
    ".Trash",
    "Trash",
    ".cache",          # we already cover ~/.cache/huggingface explicitly
    "site-packages",
})


def default_skip_dirs(home: Path) -> set[Path]:
    """Resolved paths that the stray scan should never enter."""
    candidates = (
        home / ".cache" / "huggingface",
        home / ".ollama",
        home / "LM Studio",
        home / ".lmstudio",
        home / "Library",
        home / ".Trash",
    )
    skips: set[Path] = set()
    for candidate in candidates:
        if candidate.exists():
            try:
                skips.add(candidate.resolve())
            except OSError:
                pass
    return skips


def iter_stray(
    roots: Iterable[Path],
    skip_dirs: Iterable[Path] = (),
) -> Iterator[ModelEntry]:
    """Yield stray ModelEntry results found under any of `roots`.

    Symlinks are never followed (this is what keeps traversal from escaping
    the home directory — see the project README). We additionally skip any
    directory whose resolved path is in skip_dirs or whose name is in
    NOISE_DIR_NAMES.
    """
    skip_resolved: set[Path] = set()
    for skip in skip_dirs:
        try:
            skip_resolved.add(Path(skip).resolve())
        except OSError:
            continue

    seen_files: set[Path] = set()

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            root_resolved = root.resolve()
        except OSError:
            continue

        # os.walk does not follow symlinks by default. We additionally prune
        # subdirectories in-place so we never even stat their contents.
        for dirpath, dirnames, filenames in os.walk(root_resolved, followlinks=False):
            current = Path(dirpath)

            kept: list[str] = []
            for d in dirnames:
                if d in NOISE_DIR_NAMES:
                    continue
                child = current / d
                # Skip symlinks outright (don't follow them anywhere) and
                # bail on any path that resolves outside of $HOME.
                if child.is_symlink():
                    continue
                try:
                    resolved_child = child.resolve()
                except OSError:
                    continue
                if resolved_child in skip_resolved:
                    continue
                kept.append(d)
            dirnames[:] = kept

            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext not in STRAY_EXTS:
                    continue
                file = current / fname
                # Don't follow file symlinks; only count concrete files we
                # haven't already accounted for.
                if file.is_symlink():
                    continue
                try:
                    resolved_file = file.resolve()
                except OSError:
                    continue
                if resolved_file in seen_files:
                    continue
                # If the resolved file lives under a skip dir, drop it.
                if any(_is_within(resolved_file, s) for s in skip_resolved):
                    continue
                seen_files.add(resolved_file)
                try:
                    size = resolved_file.stat().st_size
                except OSError:
                    continue
                yield ModelEntry(
                    source=SOURCE_OTHER,
                    name=str(file),
                    format=format_for_path(file),
                    size_bytes=size,
                    path=str(file),
                )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
