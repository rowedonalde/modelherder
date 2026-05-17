import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from modelherder import cli
from modelherder.models import (
    ModelEntry,
    SOURCE_HUGGINGFACE,
    SOURCE_LMSTUDIO,
    SOURCE_OLLAMA,
)


FAKE_HOME = Path("/home/fakeuser")


def _make_fake_scanner_class(
    entries: list[ModelEntry], counter: dict, key: str
):
    """Build a fake scanner class for HF / Ollama / LM Studio.

    Bumps counter[key] only when .scan() is called — not when the class is
    instantiated. The CLI constructs these scanners unconditionally so it can
    call collect_known_paths() during the stray scan, so counting in __init__
    would over-count.
    """
    entries_copy = list(entries)

    class _FakeScanner:
        def __init__(self, *args, **kwargs):
            pass

        def scan(self):
            counter[key] += 1
            return list(entries_copy)

        def collect_known_paths(self):
            return set()

    return _FakeScanner


def _make_fake_stray_scanner_class(
    entries: list[ModelEntry], raise_after: int | None = None
):
    """Build a fake StrayScanner class.

    Captures the (roots, skip_dirs) it was constructed with, and yields the
    pre-configured entries from scan(). Optionally raises KeyboardInterrupt
    partway through to exercise the cancel path.
    """
    calls: list[tuple[list[Path], set[Path]]] = []

    class _FakeStrayScanner:
        def __init__(self, roots, skip_dirs):
            calls.append((list(roots), set(skip_dirs)))

        def scan(self):
            for i, entry in enumerate(entries):
                if raise_after is not None and i == raise_after:
                    raise KeyboardInterrupt
                yield entry

    return _FakeStrayScanner, calls


class _RunResult:
    def __init__(self, exit_code: int, stdout: str, stderr: str) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


def _run_main(argv: list[str]) -> _RunResult:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        rc = cli.main(argv)
    return _RunResult(rc, stdout.getvalue(), stderr.getvalue())


class ParseArgsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        args = cli._parse_args([])
        self.assertIsNone(args.sources)
        self.assertIsNone(args.format_filter)
        self.assertFalse(args.json)
        self.assertFalse(args.no_stray)
        self.assertEqual(args.path, [])

    def test_sources_split_and_lowercased(self) -> None:
        args = cli._parse_args(["--sources", "ollama,huggingface"])
        self.assertEqual(args.sources, ["ollama", "huggingface"])

    def test_sources_whitespace_and_case_normalised(self) -> None:
        args = cli._parse_args(["--sources", " Ollama , LMSTUDIO "])
        self.assertEqual(args.sources, ["ollama", "lmstudio"])

    def test_sources_empty_string_yields_empty_list(self) -> None:
        args = cli._parse_args(["--sources", ""])
        self.assertEqual(args.sources, [])

    def test_format_filter_kept_raw(self) -> None:
        # Normalisation (case + leading dot) happens in _filter_format, not here.
        args = cli._parse_args(["--format", ".GGUF"])
        self.assertEqual(args.format_filter, ".GGUF")

    def test_json_and_no_stray_flags(self) -> None:
        args = cli._parse_args(["--json", "--no-stray"])
        self.assertTrue(args.json)
        self.assertTrue(args.no_stray)

    def test_path_appends(self) -> None:
        args = cli._parse_args(["--path", "/a", "--path", "/b"])
        self.assertEqual(args.path, ["/a", "/b"])


class FilterFormatTests(unittest.TestCase):
    def _entries(self) -> list[ModelEntry]:
        return [
            ModelEntry(SOURCE_OLLAMA, "a", "GGUF", 1, "/a"),
            ModelEntry(SOURCE_HUGGINGFACE, "b", "safetensors", 2, "/b"),
            ModelEntry(SOURCE_LMSTUDIO, "c", "GGUF", 3, "/c"),
        ]

    def test_none_returns_input_unchanged(self) -> None:
        entries = self._entries()
        self.assertIs(cli._filter_format(entries, None), entries)

    def test_empty_string_returns_input_unchanged(self) -> None:
        entries = self._entries()
        self.assertIs(cli._filter_format(entries, ""), entries)

    def test_matches_case_insensitively(self) -> None:
        entries = self._entries()
        result = cli._filter_format(entries, "gguf")
        self.assertEqual([e.name for e in result], ["a", "c"])

    def test_leading_dot_is_stripped(self) -> None:
        entries = self._entries()
        result = cli._filter_format(entries, ".GGUF")
        self.assertEqual([e.name for e in result], ["a", "c"])

    def test_no_matches_returns_empty(self) -> None:
        entries = self._entries()
        self.assertEqual(cli._filter_format(entries, "onnx"), [])


class MainTests(unittest.TestCase):
    """End-to-end main() tests with scanner seams replaced by fakes."""

    def _patches(
        self,
        *,
        ollama: list[ModelEntry] | None = None,
        hf: list[ModelEntry] | None = None,
        lms: list[ModelEntry] | None = None,
        stray: list[ModelEntry] | None = None,
        raise_after: int | None = None,
    ):
        """Build the fake classes main() will see."""
        self.scan_calls = {"ollama": 0, "hf": 0, "lms": 0}
        stray_cls, stray_calls = _make_fake_stray_scanner_class(
            stray or [], raise_after=raise_after
        )
        fakes = {
            "ollama_cls": _make_fake_scanner_class(
                ollama or [], self.scan_calls, "ollama"
            ),
            "hf_cls": _make_fake_scanner_class(
                hf or [], self.scan_calls, "hf"
            ),
            "lms_cls": _make_fake_scanner_class(
                lms or [], self.scan_calls, "lms"
            ),
            "stray_cls": stray_cls,
            "stray_calls": stray_calls,
        }
        return contextlib.ExitStack(), fakes

    def _apply(self, stack, fakes):
        stack.enter_context(patch.object(cli, "OllamaScanner", new=fakes["ollama_cls"]))
        stack.enter_context(patch.object(cli, "HuggingFaceScanner", new=fakes["hf_cls"]))
        stack.enter_context(patch.object(cli, "LMStudioScanner", new=fakes["lms_cls"]))
        stack.enter_context(patch.object(cli, "StrayScanner", new=fakes["stray_cls"]))
        stack.enter_context(patch.object(cli, "default_skip_dirs", new=lambda home: set()))
        # Pin Path.home() for deterministic --path / scan_roots assertions.
        stack.enter_context(patch.object(cli.Path, "home", classmethod(lambda cls: FAKE_HOME)))

    def test_default_invocation_calls_all_scanners_and_returns_zero(self) -> None:
        stack, fakes = self._patches()
        with stack:
            self._apply(stack, fakes)
            result = _run_main([])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(self.scan_calls, {"ollama": 1, "hf": 1, "lms": 1})
        self.assertEqual(len(fakes["stray_calls"]), 1)
        # No entries → the "no model files found" message goes to stdout.
        self.assertIn("No model files found.", result.stdout)

    def test_no_stray_skips_stray_scan(self) -> None:
        stack, fakes = self._patches()
        with stack:
            self._apply(stack, fakes)
            result = _run_main(["--no-stray"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(fakes["stray_calls"], [])

    def test_sources_subset_runs_only_selected_scanners(self) -> None:
        stack, fakes = self._patches()
        with stack:
            self._apply(stack, fakes)
            _run_main(["--sources", "ollama", "--no-stray"])
        self.assertEqual(self.scan_calls, {"ollama": 1, "hf": 0, "lms": 0})

    def test_unknown_source_returns_two_and_skips_scanners(self) -> None:
        stack, fakes = self._patches()
        with stack:
            self._apply(stack, fakes)
            result = _run_main(["--sources", "nope"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("unknown source(s): nope", result.stderr)
        self.assertEqual(self.scan_calls, {"ollama": 0, "hf": 0, "lms": 0})
        self.assertEqual(fakes["stray_calls"], [])

    def test_json_output_is_parseable_and_on_stdout(self) -> None:
        entry = ModelEntry(SOURCE_OLLAMA, "llama3:8b", "GGUF", 1024, "/tmp/m.gguf")
        stack, fakes = self._patches(ollama=[entry])
        with stack:
            self._apply(stack, fakes)
            result = _run_main(["--json", "--no-stray"])
        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.stdout)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["name"], "llama3:8b")
        # Stderr should not also contain the JSON payload.
        self.assertNotIn("llama3:8b", result.stderr)

    def test_format_filter_drops_non_matching_entries(self) -> None:
        entries = [
            ModelEntry(SOURCE_OLLAMA, "keep", "GGUF", 1, "/a"),
            ModelEntry(SOURCE_OLLAMA, "drop", "safetensors", 2, "/b"),
        ]
        stack, fakes = self._patches(ollama=entries)
        with stack:
            self._apply(stack, fakes)
            result = _run_main(["--json", "--no-stray", "--format", "gguf"])
        parsed = json.loads(result.stdout)
        self.assertEqual([e["name"] for e in parsed], ["keep"])

    def test_keyboard_interrupt_during_stray_keeps_partial_results(self) -> None:
        kept = ModelEntry("other", "stray-a", "GGUF", 1, "/tmp/a.gguf")
        # raise_after=1 → yields the first entry, then raises before the second.
        stack, fakes = self._patches(
            stray=[
                kept,
                ModelEntry("other", "stray-b", "GGUF", 2, "/tmp/b.gguf"),
            ],
            raise_after=1,
        )
        with stack:
            self._apply(stack, fakes)
            result = _run_main(["--json"])
        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.stdout)
        self.assertEqual([e["name"] for e in parsed], ["stray-a"])
        self.assertIn("Stray scan cancelled", result.stderr)

    def test_extra_path_is_forwarded_to_stray_scanner(self) -> None:
        stack, fakes = self._patches()
        with stack:
            self._apply(stack, fakes)
            _run_main(["--path", "/tmp/extra"])
        self.assertEqual(len(fakes["stray_calls"]), 1)
        roots, _ = fakes["stray_calls"][0]
        self.assertIn(FAKE_HOME, roots)
        self.assertIn(Path("/tmp/extra"), roots)
