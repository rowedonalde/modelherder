import unittest
from pathlib import Path

from modelherder.models import SOURCE_HUGGINGFACE
from modelherder.scanners.huggingface import HuggingFaceScanner


FIXTURES = Path(__file__).parent / "fakemodels" / "huggingface" / "hub"


class HuggingFaceScannerTests(unittest.TestCase):
    def test_scan_returns_entries_for_each_model(self) -> None:
        entries = HuggingFaceScanner(hub_dir=FIXTURES).scan()
        by_name = {e.name: e for e in entries}
        self.assertIn("meta-llama/Llama-3.2-1B", by_name)
        self.assertIn("openai/whisper", by_name)

    def test_entries_are_tagged_as_huggingface_source(self) -> None:
        entries = HuggingFaceScanner(hub_dir=FIXTURES).scan()
        self.assertTrue(entries)
        for entry in entries:
            self.assertEqual(entry.source, SOURCE_HUGGINGFACE)

    def test_format_label_comes_from_file_extension(self) -> None:
        entries = HuggingFaceScanner(hub_dir=FIXTURES).scan()
        by_name = {e.name: e for e in entries}
        self.assertEqual(by_name["meta-llama/Llama-3.2-1B"].format, "safetensors")
        self.assertEqual(by_name["openai/whisper"].format, "bin")

    def test_size_is_sum_of_blob_bytes(self) -> None:
        entries = HuggingFaceScanner(hub_dir=FIXTURES).scan()
        by_name = {e.name: e for e in entries}
        # model.safetensors is "xx" (2 bytes); pytorch_model.bin is "xxx" (3 bytes)
        self.assertEqual(by_name["meta-llama/Llama-3.2-1B"].size_bytes, 2)
        self.assertEqual(by_name["openai/whisper"].size_bytes, 3)

    def test_path_points_at_snapshot_dir(self) -> None:
        entries = HuggingFaceScanner(hub_dir=FIXTURES).scan()
        by_name = {e.name: e for e in entries}
        self.assertTrue(by_name["meta-llama/Llama-3.2-1B"].path.endswith("/snapshots/abc123"))

    def test_missing_hub_dir_returns_empty_list(self) -> None:
        entries = HuggingFaceScanner(hub_dir=FIXTURES / "does-not-exist").scan()
        self.assertEqual(entries, [])

    def test_collect_known_paths_includes_hub_dir(self) -> None:
        paths = HuggingFaceScanner(hub_dir=FIXTURES).collect_known_paths()
        self.assertIn(FIXTURES.resolve(), paths)

    def test_collect_known_paths_empty_when_hub_missing(self) -> None:
        paths = HuggingFaceScanner(hub_dir=FIXTURES / "missing").collect_known_paths()
        self.assertEqual(paths, set())


if __name__ == "__main__":
    unittest.main()
