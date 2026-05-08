# CLAUDE.md

Guidance for Claude Code working in this repo. The product spec lives in
[`INSTRUCTIONS.md`](./INSTRUCTIONS.md) — read it before adding features or
changing scanner behaviour.

## What this is

`modelherder` is a CLI that inventories generative AI model files on disk.
It has source-specific scanners for Ollama / HuggingFace / LM Studio plus a
generic recursive stray scan. Output is either a Rich-grouped table or JSON.

## Layout

```
src/modelherder/
  cli.py          # argparse, source filtering, spinner + Ctrl-C handling
  models.py       # ModelEntry, source constants, MODEL_EXTS vs STRAY_EXTS
  output.py       # Rich tables (grouped per source) + JSON renderer
  sizes.py        # human-readable byte formatting
  scanners/
    huggingface.py
    ollama.py
    lmstudio.py
    stray.py
```

`main.py` at the repo root is a thin shim so `uv run main.py` still works.
Real entry points are `uv run modelherder` and `python -m modelherder`.

## Running and developing

- Install / sync deps: `uv sync`
- Run: `uv run modelherder [flags]`
- Add a dependency: `uv add <pkg>` (do not hand-edit `pyproject.toml` for
  deps unless you have a reason)
- The console script `modelherder` is registered via `[project.scripts]`

### Testing

If adding tests, use `unittest` (project preference, not `pytest`). Place
them under `tests/` and run with `uv run python -m unittest`.

## Non-obvious design decisions

These are the things that aren't derivable from reading the code in
isolation — change them only with intent.

### Two extension sets

`models.py` defines both `MODEL_EXTS` (used by HF and LM Studio) and
`STRAY_EXTS` (used by the recursive stray walk). `STRAY_EXTS` deliberately
omits `.bin` because the broad walk hits Photos library caches, Go module
testdata, and macOS app resources that all use `.bin`. The HF and LM Studio
scanners are already scoped to model directories, so `.bin` is safe (and
useful — many HF models ship `pytorch_model.bin`).

If you broaden either set, sanity-check the stray output on a real machine.

### Stray scan never follows symlinks

`stray.py` uses `os.walk(followlinks=False)` and additionally skips any
symlinked file/directory child outright. This is intentional: it's what
keeps the walk from escaping `$HOME` (per the spec) without needing a
separate "is this path under home?" check, which would also break the
`--path` flag for directories outside home.

If you ever want to follow symlinks within home, you'll need to reintroduce
a containment check and handle cycles — don't just flip `followlinks`.

### HF aggregation per (model, format)

The HF scanner groups files by `(model_name, format_label)` and sums unique
blob sizes. This collapses sharded safetensors checkpoints
(`model-00001-of-00007.safetensors`, ...) into one row. De-duplication is
keyed on the resolved blob path, not the symlink path, so multiple
snapshots pointing at the same blob count once.

### Ollama display names

Manifests live at `manifests/{registry}/{namespace}/{name}/{tag}`. The
display logic in `_display_name`:

- `registry.ollama.ai` + `library` namespace → `name:tag` (e.g. `gemma3:4b`)
- `registry.ollama.ai` + other namespace → `namespace/name:tag`
- Other registries (e.g. `hf.co`) → `host/.../name:tag`

The model layer is identified by `mediaType ==
"application/vnd.ollama.image.model"`. Manifests record digests as
`sha256:abc...` but blobs on disk are named `sha256-abc...` — the scanner
swaps the separator.

### Output streams

JSON and the table both go to stdout. The Rich spinner / status message and
the "scan cancelled" notice go to stderr (`Console(stderr=True)`), so
piping `--json` to `jq` works without contamination.

### Ctrl-C during stray scan

`cli.py` wraps the stray loop in `try/except KeyboardInterrupt`, accumulating
into a list as entries are yielded so partial results survive a cancel.
Don't refactor the scanner to return a fully-realized list — the generator
shape is what makes Ctrl-C produce meaningful output.

## Things to avoid

- Don't reintroduce a "must be under $HOME" filter in `iter_stray`. It
  breaks `--path` and is redundant with not following symlinks.
- Don't auto-follow HF snapshot symlinks via `Path.resolve()` in a way that
  loses the snapshot path — output uses snapshot paths intentionally
  (the user wants to know which snapshot).
- Don't add scanners that shell out — keep everything pure-Python so a bare
  `uv sync` is enough to run the tool.
