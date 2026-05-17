import io
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from modelherder import output
from modelherder.models import (
    ModelEntry,
    SOURCE_HUGGINGFACE,
    SOURCE_LMSTUDIO,
    SOURCE_OLLAMA,
    SOURCE_OTHER,
)
from modelherder.output import _shorten, render_json, render_table
from modelherder.sizes import human_size


FAKE_HOME = Path("/home/fakeuser")


def _make_console() -> Console:
    """Return a Console that writes to a captured StringIO.

    Width is pinned so column wrapping doesn't make assertions flaky.
    """
    return Console(
        file=io.StringIO(),
        width=200,
        force_terminal=False,
        no_color=True,
    )


class ShortenTests(unittest.TestCase):
    def test_path_under_home_gets_tilde(self) -> None:
        with patch.object(output, "HOME", FAKE_HOME):
            self.assertEqual(
                _shorten("/home/fakeuser/Models/foo.gguf"),
                os.path.join("~", "Models/foo.gguf"),
            )

    def test_path_outside_home_returned_unchanged(self) -> None:
        with patch.object(output, "HOME", FAKE_HOME):
            self.assertEqual(_shorten("/var/tmp/foo.gguf"), "/var/tmp/foo.gguf")

    def test_home_itself_returns_tilde_dot(self) -> None:
        # Path.relative_to(HOME) returns Path("."), so os.path.join("~", ".")
        # yields "~/.". Documents current behaviour.
        with patch.object(output, "HOME", FAKE_HOME):
            self.assertEqual(_shorten(str(FAKE_HOME)), os.path.join("~", "."))


class RenderJsonTests(unittest.TestCase):
    def test_empty_list_is_valid_empty_json(self) -> None:
        self.assertEqual(json.loads(render_json([])), [])

    def test_single_entry_has_expected_keys_and_size_human(self) -> None:
        entry = ModelEntry(
            source=SOURCE_OLLAMA,
            name="llama3:8b",
            format="GGUF",
            size_bytes=2 * 1024 * 1024 * 1024,
            path="/tmp/foo.gguf",
        )
        parsed = json.loads(render_json([entry]))
        self.assertEqual(len(parsed), 1)
        self.assertEqual(
            parsed[0],
            {
                "source": "ollama",
                "name": "llama3:8b",
                "format": "GGUF",
                "size_bytes": 2 * 1024 * 1024 * 1024,
                "size_human": human_size(2 * 1024 * 1024 * 1024),
                "path": "/tmp/foo.gguf",
            },
        )

    def test_multiple_entries_preserve_order(self) -> None:
        entries = [
            ModelEntry(SOURCE_OLLAMA, "a", "GGUF", 1, "/a"),
            ModelEntry(SOURCE_HUGGINGFACE, "b", "safetensors", 2, "/b"),
            ModelEntry(SOURCE_LMSTUDIO, "c", "GGUF", 3, "/c"),
        ]
        parsed = json.loads(render_json(entries))
        self.assertEqual([e["name"] for e in parsed], ["a", "b", "c"])

    def test_output_is_indented(self) -> None:
        entry = ModelEntry(SOURCE_OLLAMA, "a", "GGUF", 1, "/a")
        raw = render_json([entry])
        # indent=2 produces a newline followed by two spaces somewhere.
        self.assertIn("\n  ", raw)


class RenderTableTests(unittest.TestCase):
    def _render(self, entries: list[ModelEntry]) -> str:
        console = _make_console()
        render_table(entries, console)
        assert isinstance(console.file, io.StringIO)
        return console.file.getvalue()

    def test_empty_entries_prints_no_models_message(self) -> None:
        out = self._render([])
        self.assertIn("No model files found.", out)

    def test_single_ollama_entry_renders_header_name_format_size(self) -> None:
        entry = ModelEntry(
            source=SOURCE_OLLAMA,
            name="llama3:8b",
            format="GGUF",
            size_bytes=4 * 1024 * 1024 * 1024,
            path="/home/fakeuser/.ollama/blobs/abc",
        )
        with patch.object(output, "HOME", FAKE_HOME):
            out = self._render([entry])
        self.assertIn("OLLAMA", out)
        self.assertIn("llama3:8b", out)
        self.assertIn("GGUF", out)
        self.assertIn(human_size(entry.size_bytes), out)

    def test_source_other_uses_path_format_size_columns_only(self) -> None:
        entry = ModelEntry(
            source=SOURCE_OTHER,
            name="ignored-for-other",
            format="GGUF",
            size_bytes=1024,
            path="/home/fakeuser/strays/foo.gguf",
        )
        with patch.object(output, "HOME", FAKE_HOME):
            out = self._render([entry])
        self.assertIn("OTHER", out)
        # For SOURCE_OTHER the column order is Path, Format, Size — no Model.
        path_idx = out.find("Path")
        format_idx = out.find("Format")
        self.assertNotEqual(path_idx, -1)
        self.assertNotEqual(format_idx, -1)
        self.assertLess(path_idx, format_idx)
        # The "Model" column header should not appear in a SOURCE_OTHER-only
        # render.
        self.assertNotIn("Model", out)

    def test_mixed_sources_render_in_canonical_order(self) -> None:
        entries = [
            # Intentionally not in canonical order — render_table should still
            # emit OLLAMA, HUGGINGFACE, LM STUDIO, OTHER.
            ModelEntry(SOURCE_LMSTUDIO, "lms-model", "GGUF", 10, "/tmp/lms"),
            ModelEntry(SOURCE_OLLAMA, "oll-model", "GGUF", 10, "/tmp/oll"),
            ModelEntry(SOURCE_OTHER, "stray", "GGUF", 10, "/tmp/stray"),
            ModelEntry(SOURCE_HUGGINGFACE, "hf-model", "safetensors", 10, "/tmp/hf"),
        ]
        out = self._render(entries)
        ollama_idx = out.find("OLLAMA")
        hf_idx = out.find("HUGGINGFACE")
        lms_idx = out.find("LM STUDIO")
        other_idx = out.find("OTHER")
        self.assertGreater(ollama_idx, -1)
        self.assertGreater(hf_idx, ollama_idx)
        self.assertGreater(lms_idx, hf_idx)
        self.assertGreater(other_idx, lms_idx)

    def test_empty_source_is_omitted(self) -> None:
        # Only an ollama entry — no HUGGINGFACE / LM STUDIO / OTHER headers
        # should appear.
        entry = ModelEntry(SOURCE_OLLAMA, "only", "GGUF", 1, "/tmp/x")
        out = self._render([entry])
        self.assertIn("OLLAMA", out)
        self.assertNotIn("HUGGINGFACE", out)
        self.assertNotIn("LM STUDIO", out)
        self.assertNotIn("OTHER", out)

    def test_section_total_row_is_rendered(self) -> None:
        entries = [
            ModelEntry(SOURCE_OLLAMA, "a", "GGUF", 1024, "/tmp/a"),
            ModelEntry(SOURCE_OLLAMA, "b", "GGUF", 2048, "/tmp/b"),
        ]
        out = self._render(entries)
        self.assertIn("TOTAL", out)
        self.assertIn(human_size(1024 + 2048), out)

    def test_section_total_sums_bytes_not_human_strings(self) -> None:
        # 1500 + 1500 bytes = 3000 bytes => "2.9 KiB" via human_size.
        # If we naively added human-readable strings ("1.5 KiB" + "1.5 KiB"),
        # we'd get "3.0 KiB" — make sure we don't do that.
        entries = [
            ModelEntry(SOURCE_OLLAMA, "a", "GGUF", 1500, "/tmp/a"),
            ModelEntry(SOURCE_OLLAMA, "b", "GGUF", 1500, "/tmp/b"),
        ]
        out = self._render(entries)
        self.assertIn(human_size(3000), out)

    def test_source_other_also_has_total_row(self) -> None:
        entries = [
            ModelEntry(SOURCE_OTHER, "x", "GGUF", 4096, "/tmp/x.gguf"),
            ModelEntry(SOURCE_OTHER, "y", "safetensors", 8192, "/tmp/y.safetensors"),
        ]
        out = self._render(entries)
        self.assertIn("TOTAL", out)
        self.assertIn(human_size(4096 + 8192), out)

    def test_grand_total_reflects_sum_across_sections(self) -> None:
        entries = [
            ModelEntry(SOURCE_OLLAMA, "a", "GGUF", 100, "/a"),
            ModelEntry(SOURCE_HUGGINGFACE, "b", "safetensors", 200, "/b"),
            ModelEntry(SOURCE_LMSTUDIO, "c", "GGUF", 300, "/c"),
            ModelEntry(SOURCE_OTHER, "d", "GGUF", 400, "/d"),
        ]
        out = self._render(entries)
        self.assertIn(human_size(1000), out)
        # TOTAL appears once per non-empty section plus once as a grand total.
        self.assertGreaterEqual(out.count("TOTAL"), 5)

    def test_grand_total_present_with_single_section(self) -> None:
        entry = ModelEntry(SOURCE_OLLAMA, "only", "GGUF", 12345, "/tmp/x")
        out = self._render([entry])
        # Section total and grand total both equal 12345 bytes; ensure
        # the formatted value appears at least twice.
        self.assertGreaterEqual(out.count(human_size(12345)), 2)
