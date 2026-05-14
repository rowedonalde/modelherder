from __future__ import annotations

import json
import os
from pathlib import Path

from rich.console import Console
from rich.table import Table

from modelherder.models import (
    ALL_SOURCES,
    ModelEntry,
    SOURCE_HEADERS,
    SOURCE_OTHER,
)
from modelherder.sizes import human_size


HOME = Path.home()


def _shorten(path_str: str) -> str:
    """Replace a leading $HOME with ~ for readability."""
    try:
        p = Path(path_str)
        rel = p.relative_to(HOME)
        return os.path.join("~", str(rel))
    except ValueError:
        return path_str


def render_table(entries: list[ModelEntry], console: Console) -> None:
    """Print one Rich table per source, in the canonical source ordering.

    Each section ends with a TOTAL row summing the bytes for that section,
    and a final grand TOTAL is printed after the last table. Totals are
    computed by summing raw byte counts and then formatting once — never
    by adding human-readable strings together.
    """
    grouped: dict[str, list[ModelEntry]] = {s: [] for s in ALL_SOURCES}
    for entry in entries:
        grouped.setdefault(entry.source, []).append(entry)

    if not entries:
        console.print("[dim]No model files found.[/dim]")
        return

    grand_total_bytes = 0
    for source in ALL_SOURCES:
        rows = grouped.get(source) or []
        if not rows:
            continue
        header = SOURCE_HEADERS.get(source, source.upper())
        table = Table(
            title=f"[bold]{header}[/bold]",
            title_justify="left",
            show_lines=False,
            pad_edge=False,
        )
        section_total_bytes = 0
        if source == SOURCE_OTHER:
            # Stray entries don't have a useful "name" distinct from the path
            table.add_column("Path", overflow="fold")
            table.add_column("Format")
            table.add_column("Size", justify="right")
            for r in rows:
                table.add_row(_shorten(r.path), r.format, human_size(r.size_bytes))
                section_total_bytes += r.size_bytes
            table.add_section()
            table.add_row(
                "[bold]TOTAL[/bold]",
                "",
                f"[bold]{human_size(section_total_bytes)}[/bold]",
            )
        else:
            table.add_column("Model", overflow="fold")
            table.add_column("Format")
            table.add_column("Size", justify="right")
            table.add_column("Path", overflow="fold")
            for r in rows:
                table.add_row(
                    r.name,
                    r.format,
                    human_size(r.size_bytes),
                    _shorten(r.path),
                )
                section_total_bytes += r.size_bytes
            table.add_section()
            table.add_row(
                "[bold]TOTAL[/bold]",
                "",
                f"[bold]{human_size(section_total_bytes)}[/bold]",
                "",
            )
        grand_total_bytes += section_total_bytes
        console.print(table)

    console.print("[bold]TOTAL[/bold]")
    console.print(f"[bold]{human_size(grand_total_bytes)}[/bold]")


def render_json(entries: list[ModelEntry]) -> str:
    payload = [
        {
            "source": e.source,
            "name": e.name,
            "format": e.format,
            "size_bytes": e.size_bytes,
            "size_human": human_size(e.size_bytes),
            "path": e.path,
        }
        for e in entries
    ]
    return json.dumps(payload, indent=2)
