import unittest
from pathlib import Path

from modelherder.models import SOURCE_OLLAMA
from modelherder.scanners.ollama import OllamaScanner


FIXTURES = Path(__file__).parent / "fakemodels" / "ollama"


class OllamaScannerTests(unittest.TestCase):
    def test_scan_finds_library_models_with_short_name(self) -> None:
        entries = OllamaScanner(root=FIXTURES).scan()
        names = {e.name for e in entries}
        # Library namespace under registry.ollama.ai should be stripped.
        self.assertIn("gemma3:4b", names)
        self.assertIn("llama3:8b", names)

    def test_scan_keeps_hf_registry_prefix_in_name(self) -> None:
        entries = OllamaScanner(root=FIXTURES).scan()
        names = {e.name for e in entries}
        # Non-ollama registries keep their host and namespace.
        self.assertIn("hf.co/myorg/myrepo:Q4_K_M", names)

    def test_entries_are_tagged_as_ollama_source_and_gguf_format(self) -> None:
        entries = OllamaScanner(root=FIXTURES).scan()
        self.assertTrue(entries)
        for entry in entries:
            self.assertEqual(entry.source, SOURCE_OLLAMA)
            self.assertEqual(entry.format, "GGUF")

    def test_size_comes_from_manifest_layer(self) -> None:
        entries = OllamaScanner(root=FIXTURES).scan()
        by_name = {e.name: e for e in entries}
        # Sizes match what the manifests declare in the model-layer entry.
        self.assertEqual(by_name["gemma3:4b"].size_bytes, 4000000)
        self.assertEqual(by_name["llama3:8b"].size_bytes, 8000000)
        self.assertEqual(by_name["hf.co/myorg/myrepo:Q4_K_M"].size_bytes, 2500000)

    def test_path_points_at_blob_with_dash_separator(self) -> None:
        entries = OllamaScanner(root=FIXTURES).scan()
        by_name = {e.name: e for e in entries}
        # Manifest records digests with `sha256:` but on-disk blobs use `sha256-`.
        path = by_name["gemma3:4b"].path
        self.assertIn("blobs/sha256-aaaaaaaa", path)

    def test_missing_root_returns_empty_list(self) -> None:
        entries = OllamaScanner(root=FIXTURES / "does-not-exist").scan()
        self.assertEqual(entries, [])

    def test_collect_known_paths_includes_root(self) -> None:
        paths = OllamaScanner(root=FIXTURES).collect_known_paths()
        self.assertIn(FIXTURES.resolve(), paths)

    def test_collect_known_paths_empty_when_root_missing(self) -> None:
        paths = OllamaScanner(root=FIXTURES / "missing").collect_known_paths()
        self.assertEqual(paths, set())


if __name__ == "__main__":
    unittest.main()
