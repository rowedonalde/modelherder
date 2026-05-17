import unittest
from pathlib import Path

from modelherder.models import (
    ALL_SOURCES,
    FORMAT_BY_EXT,
    MODEL_EXTS,
    ModelEntry,
    SOURCE_HUGGINGFACE,
    SOURCE_LMSTUDIO,
    SOURCE_OLLAMA,
    SOURCE_OTHER,
    STRAY_EXTS,
    format_for_path,
)


class FormatForPathTests(unittest.TestCase):
    def test_known_extensions(self) -> None:
        for ext, label in FORMAT_BY_EXT.items():
            with self.subTest(ext=ext):
                self.assertEqual(format_for_path(Path(f"weights{ext}")), label)

    def test_mixed_case_extension_normalises_to_lowercase(self) -> None:
        self.assertEqual(format_for_path(Path("model.GGUF")), "GGUF")
        self.assertEqual(format_for_path(Path("model.SafeTensors")), "safetensors")

    def test_unknown_extension_returns_suffix_without_dot(self) -> None:
        self.assertEqual(format_for_path(Path("foo.weights")), "weights")

    def test_no_extension_returns_unknown(self) -> None:
        self.assertEqual(format_for_path(Path("README")), "unknown")

    def test_dotfile_with_no_real_suffix_returns_unknown(self) -> None:
        # Path(".env").suffix is "" — documents the current behaviour.
        self.assertEqual(format_for_path(Path(".env")), "unknown")


class ModelEntryTests(unittest.TestCase):
    def test_as_dict_round_trips_all_fields(self) -> None:
        entry = ModelEntry(
            source=SOURCE_OLLAMA,
            name="llama3:8b",
            format="GGUF",
            size_bytes=1234567,
            path="/tmp/foo.gguf",
        )
        self.assertEqual(
            entry.as_dict(),
            {
                "source": "ollama",
                "name": "llama3:8b",
                "format": "GGUF",
                "size_bytes": 1234567,
                "path": "/tmp/foo.gguf",
            },
        )


class ConstantsTests(unittest.TestCase):
    def test_stray_exts_excludes_bin(self) -> None:
        # Non-obvious invariant called out in CLAUDE.md: STRAY_EXTS must not
        # include .bin (too many false positives in the recursive walk).
        self.assertNotIn(".bin", STRAY_EXTS)

    def test_model_exts_includes_bin(self) -> None:
        # HF and LM Studio scanners are already scoped — .bin is useful there.
        self.assertIn(".bin", MODEL_EXTS)

    def test_all_sources_canonical_ordering(self) -> None:
        # output.render_table iterates ALL_SOURCES to render tables in order.
        self.assertEqual(
            ALL_SOURCES,
            (SOURCE_OLLAMA, SOURCE_HUGGINGFACE, SOURCE_LMSTUDIO, SOURCE_OTHER),
        )
