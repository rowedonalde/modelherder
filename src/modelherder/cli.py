from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from modelherder.models import (
    ModelEntry,
    SOURCE_HUGGINGFACE,
    SOURCE_LMSTUDIO,
    SOURCE_OLLAMA,
)
from modelherder.output import render_json, render_table
from modelherder.scanners import (
    default_skip_dirs,
    iter_stray,
    scan_huggingface,
    scan_lmstudio,
    scan_ollama,
)
from modelherder.scanners.huggingface import (
    collect_known_paths as hf_known_paths,
)
from modelherder.scanners.lmstudio import (
    collect_known_paths as lms_known_paths,
)
from modelherder.scanners.ollama import (
    collect_known_paths as ollama_known_paths,
)


SOURCE_CHOICES = (
    SOURCE_OLLAMA,
    SOURCE_HUGGINGFACE,
    SOURCE_LMSTUDIO,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="modelherder",
        description="Inventory generative AI model files installed on your machine.",
    )
    parser.add_argument(
        "--sources",
        type=lambda s: [x.strip().lower() for x in s.split(",") if x.strip()],
        default=None,
        help=(
            "Comma-separated list of sources to scan "
            f"(choices: {', '.join(SOURCE_CHOICES)}). "
            "Defaults to all known sources."
        ),
    )
    parser.add_argument(
        "--format",
        dest="format_filter",
        default=None,
        help="Filter results to a single file format (e.g. gguf, safetensors).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a table.",
    )
    parser.add_argument(
        "--no-stray",
        action="store_true",
        help="Skip the recursive scan for unattributed model files.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="DIR",
        help="Extra directory to include in the stray scan. May be repeated.",
    )
    return parser.parse_args(argv)


def _filter_format(entries: list[ModelEntry], fmt: str | None) -> list[ModelEntry]:
    if not fmt:
        return entries
    target = fmt.strip().lstrip(".").lower()
    return [e for e in entries if e.format.lower() == target]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    requested_sources: set[str]
    if args.sources is None:
        requested_sources = set(SOURCE_CHOICES)
    else:
        unknown = [s for s in args.sources if s not in SOURCE_CHOICES]
        if unknown:
            print(
                f"error: unknown source(s): {', '.join(unknown)}. "
                f"Valid choices: {', '.join(SOURCE_CHOICES)}.",
                file=sys.stderr,
            )
            return 2
        requested_sources = set(args.sources)

    # Rich writes the spinner to stderr; JSON output goes to stdout cleanly.
    console = Console(stderr=True)

    entries: list[ModelEntry] = []

    if SOURCE_OLLAMA in requested_sources:
        entries.extend(scan_ollama())
    if SOURCE_HUGGINGFACE in requested_sources:
        entries.extend(scan_huggingface())
    if SOURCE_LMSTUDIO in requested_sources:
        entries.extend(scan_lmstudio())

    if not args.no_stray:
        home = Path.home()
        skip_dirs = default_skip_dirs(home)
        skip_dirs |= hf_known_paths()
        skip_dirs |= ollama_known_paths()
        skip_dirs |= lms_known_paths()

        scan_roots: list[Path] = [home]
        for extra in args.path:
            scan_roots.append(Path(extra).expanduser())

        stray_entries: list[ModelEntry] = []
        try:
            with console.status(
                "[bold]Scanning for stray models...[/bold] (Ctrl-C to stop)",
                spinner="dots",
            ):
                for entry in iter_stray(scan_roots, skip_dirs):
                    stray_entries.append(entry)
        except KeyboardInterrupt:
            console.print(
                "[yellow]Stray scan cancelled — showing partial results.[/yellow]"
            )
        entries.extend(stray_entries)

    entries = _filter_format(entries, args.format_filter)

    if args.json:
        print(render_json(entries))
    else:
        # Render the table to stdout so users can pipe it.
        stdout_console = Console()
        render_table(entries, stdout_console)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
