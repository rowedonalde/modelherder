from __future__ import annotations

import json
from pathlib import Path

from modelherder.models import ModelEntry, SOURCE_OLLAMA


DEFAULT_OLLAMA_ROOT = Path("~/.ollama/models").expanduser()
MODEL_LAYER_MEDIATYPE = "application/vnd.ollama.image.model"


def _display_name(parts: tuple[str, ...]) -> str:
    """Build a human-readable model name from manifest path parts.

    parts is the relative path inside manifests/, e.g.
        ('registry.ollama.ai', 'library', 'gpt-oss', '120b')   -> 'gpt-oss:120b'
        ('registry.ollama.ai', 'someuser', 'mymodel', 'latest') -> 'someuser/mymodel:latest'
        ('hf.co', 'org', 'repo', 'Q4_K_M')                      -> 'hf.co/org/repo:Q4_K_M'
    """
    if len(parts) < 2:
        return ":".join(parts)
    *prefix, tag = parts
    if prefix and prefix[0] == "registry.ollama.ai":
        prefix = prefix[1:]
        if prefix and prefix[0] == "library":
            prefix = prefix[1:]
    return f"{'/'.join(prefix)}:{tag}"


def scan_ollama(root: Path = DEFAULT_OLLAMA_ROOT) -> list[ModelEntry]:
    if not root.exists():
        return []

    manifests_dir = root / "manifests"
    blobs_dir = root / "blobs"
    if not manifests_dir.is_dir():
        return []

    results: list[ModelEntry] = []

    for manifest_file in sorted(manifests_dir.rglob("*")):
        if not manifest_file.is_file():
            continue
        try:
            manifest = json.loads(manifest_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        layers = manifest.get("layers") or []
        model_layer = next(
            (layer for layer in layers if layer.get("mediaType") == MODEL_LAYER_MEDIATYPE),
            None,
        )
        if not model_layer:
            continue

        digest = model_layer.get("digest") or ""
        # Manifests record digests as 'sha256:abc...' but blobs on disk are
        # stored with a dash: 'sha256-abc...'.
        blob_filename = digest.replace(":", "-", 1)
        blob_path = blobs_dir / blob_filename

        size = model_layer.get("size")
        if not isinstance(size, int):
            try:
                size = blob_path.stat().st_size if blob_path.exists() else 0
            except OSError:
                size = 0

        rel_parts = manifest_file.relative_to(manifests_dir).parts
        name = _display_name(rel_parts)

        results.append(
            ModelEntry(
                source=SOURCE_OLLAMA,
                name=name,
                format="GGUF",
                size_bytes=size,
                path=str(blob_path),
            )
        )

    results.sort(key=lambda e: e.name.lower())
    return results


def collect_known_paths(root: Path = DEFAULT_OLLAMA_ROOT) -> set[Path]:
    paths: set[Path] = set()
    if not root.exists():
        return paths
    try:
        paths.add(root.resolve())
    except OSError:
        pass
    return paths
