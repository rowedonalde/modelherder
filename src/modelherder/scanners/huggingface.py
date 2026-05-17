from pathlib import Path

from modelherder.scanners.basescanner import BaseScanner
from modelherder.models import (
    MODEL_EXTS,
    ModelEntry,
    SOURCE_HUGGINGFACE,
    format_for_path,
)


DEFAULT_HF_HUB = Path("~/.cache/huggingface/hub").expanduser()


class HuggingFaceScanner(BaseScanner):
    def __init__(self, hub_dir=DEFAULT_HF_HUB):
        self.hub_dir = hub_dir
        super().__init__()

    def scan(self) -> list[ModelEntry]:
        """Inventory model files in the HuggingFace hub cache.

        Aggregates by (model, format), summing unique blob sizes so that sharded
        safetensors checkpoints show up as a single row.
        """
        if not self.hub_dir.exists():
            return []

        results: list[ModelEntry] = []

        for model_dir in sorted(self.hub_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            name = self._decode_model_name(model_dir.name)
            if name is None:
                continue
            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.is_dir():
                continue

            # Map format -> {resolved_blob_path: size} so duplicate symlinks across
            # snapshots don't double-count. We also track a representative path.
            per_format: dict[str, dict[Path, int]] = {}
            per_format_path: dict[str, Path] = {}

            for entry in snapshots_dir.rglob("*"):
                if not entry.is_file():
                    continue
                ext = entry.suffix.lower()
                if ext not in MODEL_EXTS:
                    continue
                try:
                    resolved = entry.resolve()
                    size = resolved.stat().st_size
                except OSError:
                    continue
                fmt = format_for_path(entry)
                per_format.setdefault(fmt, {})[resolved] = size
                per_format_path.setdefault(fmt, entry)

            for fmt, blob_map in per_format.items():
                total = sum(blob_map.values())
                results.append(
                    ModelEntry(
                        source=SOURCE_HUGGINGFACE,
                        name=name,
                        format=fmt,
                        size_bytes=total,
                        path=str(per_format_path[fmt].parent),
                    )
                )

        results.sort(key=lambda e: (e.name.lower(), e.format))
        return results

    def _decode_model_name(self, dir_name: str) -> str | None:
        """Convert a HF cache dir like 'models--meta-llama--Llama-3.2-1B' to 'meta-llama/Llama-3.2-1B'."""
        if not dir_name.startswith("models--"):
            return None
        rest = dir_name[len("models--"):]
        # The HF cache uses `--` to separate org and repo. Repo names can contain
        # single dashes, so split on `--` and rejoin with `/`.
        parts = rest.split("--")
        if not parts:
            return None
        return "/".join(parts)

    def collect_known_paths(self) -> set[Path]:
        """Return resolved blob and snapshot paths so the stray scan can skip them."""
        paths: set[Path] = set()
        if not self.hub_dir.exists():
            return paths
        try:
            paths.add(self.hub_dir.resolve())
        except OSError:
            pass
        return paths