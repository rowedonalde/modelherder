from __future__ import annotations

import unittest
from pathlib import Path

from modelherder.models import SOURCE_LMSTUDIO
from modelherder.scanners.lmstudio import LMStudioScanner


FIXTURE_ROOT = Path(__file__).parent / "fakemodels" / "lmstudio" / "models"


class LMStudioScannerTests(unittest.TestCase):
    def test_scan_finds_model_files_under_root(self) -> None:
        entries = LMStudioScanner(roots=(FIXTURE_ROOT,)).scan()
        names = {e.name for e in entries}
        self.assertIn("TheBloke/Mistral-7B-v0.1-GGUF/mistral-7b.Q4_K_M", names)
        self.assertIn("lmstudio-community/qwen2/qwen2-0.5b", names)

    def test_entries_are_tagged_as_lmstudio_source(self) -> None:
        entries = LMStudioScanner(roots=(FIXTURE_ROOT,)).scan()
        self.assertTrue(entries)
        for entry in entries:
            self.assertEqual(entry.source, SOURCE_LMSTUDIO)

    def test_format_label_from_extension(self) -> None:
        entries = LMStudioScanner(roots=(FIXTURE_ROOT,)).scan()
        by_name = {e.name: e for e in entries}
        self.assertEqual(
            by_name["TheBloke/Mistral-7B-v0.1-GGUF/mistral-7b.Q4_K_M"].format,
            "GGUF",
        )
        self.assertEqual(
            by_name["lmstudio-community/qwen2/qwen2-0.5b"].format,
            "safetensors",
        )

    def test_size_is_file_byte_count(self) -> None:
        entries = LMStudioScanner(roots=(FIXTURE_ROOT,)).scan()
        by_name = {e.name: e for e in entries}
        # 2 bytes for the gguf, 4 bytes for the safetensors fixture.
        self.assertEqual(
            by_name["TheBloke/Mistral-7B-v0.1-GGUF/mistral-7b.Q4_K_M"].size_bytes,
            2,
        )
        self.assertEqual(
            by_name["lmstudio-community/qwen2/qwen2-0.5b"].size_bytes,
            4,
        )

    def test_missing_root_is_silently_skipped(self) -> None:
        entries = LMStudioScanner(
            roots=(FIXTURE_ROOT, FIXTURE_ROOT / "does-not-exist"),
        ).scan()
        # The valid root still contributes; the missing one is just ignored.
        self.assertEqual(len(entries), 2)

    def test_no_valid_roots_returns_empty_list(self) -> None:
        entries = LMStudioScanner(roots=(FIXTURE_ROOT / "missing",)).scan()
        self.assertEqual(entries, [])

    def test_collect_known_paths_includes_existing_roots(self) -> None:
        paths = LMStudioScanner(
            roots=(FIXTURE_ROOT, FIXTURE_ROOT / "missing"),
        ).collect_known_paths()
        self.assertIn(FIXTURE_ROOT.resolve(), paths)
        self.assertNotIn((FIXTURE_ROOT / "missing").resolve(), paths)


if __name__ == "__main__":
    unittest.main()
