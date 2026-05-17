from pathlib import Path

from modelherder.models import (
    MODEL_EXTS,
    ModelEntry,
    SOURCE_LMSTUDIO,
    format_for_path,
)
from modelherder.scanners.basescanner import BaseScanner


DEFAULT_LMSTUDIO_ROOTS = (
    Path("~/LM Studio/models").expanduser(),
    Path("~/.lmstudio/models").expanduser(),
)


class LMStudioScanner(BaseScanner):
    def __init__(self, roots: tuple[Path, ...] = DEFAULT_LMSTUDIO_ROOTS):
        self.roots = roots
        super().__init__()

    def scan(self) -> list[ModelEntry]:
        results: list[ModelEntry] = []

        for root in self.roots:
            if not root.exists() or not root.is_dir():
                continue
            for file in root.rglob("*"):
                if not file.is_file():
                    continue
                if file.suffix.lower() not in MODEL_EXTS:
                    continue
                try:
                    size = file.stat().st_size
                except OSError:
                    continue
                results.append(
                    ModelEntry(
                        source=SOURCE_LMSTUDIO,
                        name=self._model_name(file, root),
                        format=format_for_path(file),
                        size_bytes=size,
                        path=str(file),
                    )
                )

        results.sort(key=lambda e: e.name.lower())
        return results

    def collect_known_paths(self) -> set[Path]:
        paths: set[Path] = set()
        for root in self.roots:
            if not root.exists():
                continue
            try:
                paths.add(root.resolve())
            except OSError:
                pass
        return paths

    def _model_name(self, file: Path, root: Path) -> str:
        """Build a name from the file's path relative to the LM Studio models root."""
        try:
            rel = file.relative_to(root)
        except ValueError:
            return file.stem
        rel_no_ext = rel.with_suffix("")
        return str(rel_no_ext)
