from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

# Source identifiers used across the codebase
SOURCE_OLLAMA = "ollama"
SOURCE_HUGGINGFACE = "huggingface"
SOURCE_LMSTUDIO = "lmstudio"
SOURCE_OTHER = "other"

ALL_SOURCES = (SOURCE_OLLAMA, SOURCE_HUGGINGFACE, SOURCE_LMSTUDIO, SOURCE_OTHER)

SOURCE_HEADERS = {
    SOURCE_OLLAMA: "OLLAMA",
    SOURCE_HUGGINGFACE: "HUGGINGFACE",
    SOURCE_LMSTUDIO: "LM STUDIO",
    SOURCE_OTHER: "OTHER (unattributed)",
}

# Map file extensions to display labels for the "format" column.
FORMAT_BY_EXT = {
    ".gguf": "GGUF",
    ".safetensors": "safetensors",
    ".pt": "PyTorch",
    ".pth": "PyTorch",
    ".onnx": "ONNX",
    ".bin": "bin",
}

# Used by source-specific scanners (HF, LM Studio) which are already scoped
# to known model directories. `.bin` is included here because HF models
# frequently store weights as pytorch_model.bin / *.bin.
MODEL_EXTS = tuple(FORMAT_BY_EXT.keys())

# Used by the stray recursive scan, which walks broad parts of the filesystem.
# `.bin` is intentionally excluded — it produces too many false positives
# (Photos library caches, Go module testdata, app resources) and is not in
# the project spec for stray detection.
STRAY_EXTS = (".gguf", ".safetensors", ".pt", ".pth", ".onnx")


@dataclass
class ModelEntry:
    source: str
    name: str
    format: str
    size_bytes: int
    path: str

    def as_dict(self) -> dict:
        return asdict(self)


def format_for_path(path: Path) -> str:
    """Return the human-readable format label for a model file path."""
    return FORMAT_BY_EXT.get(path.suffix.lower(), path.suffix.lstrip(".") or "unknown")
