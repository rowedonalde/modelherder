# modelherder

A CLI that scans your machine for generative AI model files and shows where
each one came from. It understands the on-disk layouts used by Ollama,
HuggingFace, and LM Studio, and resolves opaque blob hashes back to
human-readable model names. It also does an opportunistic recursive scan
from your home directory for stray model files those tools don't know about.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for environment and execution

## Install / run

```sh
uv sync
uv run modelherder
```

The package also exposes a `modelherder` console script, so once installed
into any Python environment it works as `modelherder ...` directly.

## Run tests
```sh
uv run python -m unittest
```

## Usage

Default behaviour is a grouped table of every known source plus a stray
scan from `$HOME`:

```
$ uv run modelherder
OLLAMA
gemma3:1b               GGUF       777.5 MB   ~/.ollama/models/blobs/sha256-7cd4...
gpt-oss:120b            GGUF        60.9 GB   ~/.ollama/models/blobs/sha256-6be6...
HUGGINGFACE
Qwen/Qwen3-Embedding-0.6B   safetensors    1.1 GB   ~/.cache/huggingface/hub/...
stabilityai/stable-diffusion-3.5-medium  safetensors  15.2 GB  ~/.cache/...
LM STUDIO
lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF  GGUF  4.7 GB  ~/LM Studio/models/...
OTHER (unattributed)
~/.insightface/models/buffalo_l/det_10g.onnx   ONNX   16.1 MB
```

### Flags

| Flag | Description |
| --- | --- |
| `--sources ollama,huggingface,lmstudio` | Limit scan to specific known sources. |
| `--format gguf` | Filter rows to a single file format. |
| `--json` | Emit JSON to stdout instead of the table. |
| `--no-stray` | Skip the recursive stray scan. |
| `--path DIR` | Add an extra directory to the stray scan. Repeatable. |

JSON output is suitable for piping; the spinner and status messages go to
stderr so they don't pollute structured output.

### Examples

```sh
# Just Ollama, JSON for scripting
uv run modelherder --sources ollama --json

# Only GGUF files across all sources
uv run modelherder --format gguf

# Skip the (potentially slow) stray scan
uv run modelherder --no-stray

# Include an external models directory
uv run modelherder --path /Volumes/External/models
```

## What it scans

| Source | Location |
| --- | --- |
| Ollama | `~/.ollama/models/manifests/**` (parsed) and matching blobs in `~/.ollama/models/blobs/` |
| HuggingFace | `~/.cache/huggingface/hub/models--*/snapshots/...` (symlinks resolved to blobs) |
| LM Studio | `~/LM Studio/models/` and `~/.lmstudio/models/` |
| Stray scan | Recursive walk from `$HOME` (plus any `--path` extras) for `.gguf`, `.safetensors`, `.pt`, `.pth`, `.onnx` files |

The stray scan never follows symlinks (which prevents traversal from
escaping the home directory) and prunes well-known noise like `.git`,
`node_modules`, virtualenvs, and the directories already covered by the
source-specific scanners. Press Ctrl-C to stop the stray scan early — any
results found so far are printed.

If a known-source directory doesn't exist, the scanner skips it silently.

## Notes

- HuggingFace results are aggregated per (model, format), so a sharded
  safetensors checkpoint shows as one row with the summed blob size.
- Ollama display names follow the convention `name:tag` for the default
  `registry.ollama.ai/library` namespace, and `host/path:tag` for others
  (for example `hf.co/HauhauCS/Qwen3.5-122B-...:Q4_K_M`).
- `.bin` files are recognized inside the HuggingFace and LM Studio scanners
  but intentionally excluded from the stray walk, where they generate too
  many false positives (Photos library caches, Go module testdata, app
  resources).
