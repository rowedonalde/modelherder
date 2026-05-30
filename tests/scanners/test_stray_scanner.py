import unittest
from pathlib import Path

from modelherder.models import SOURCE_OTHER
from modelherder.scanners.stray import StrayScanner


FIXTURES = Path(__file__).parent / "fakemodels" / "stray"


class StrayScannerTests(unittest.TestCase):
    def test_scan_finds_known_extension_model_files(self) -> None:
        entries = list(StrayScanner(roots=[FIXTURES], skip_dirs=set()).scan())
        paths = {Path(e.path).name for e in entries}
        self.assertIn("checkpoint.pt", paths)
        self.assertIn("notes-llm.gguf", paths)
        self.assertIn("random.safetensors", paths)

    def test_scan_skips_noise_directories_like_node_modules(self) -> None:
        entries = list(StrayScanner(roots=[FIXTURES], skip_dirs=set()).scan())
        paths = {Path(e.path).name for e in entries}
        # hidden.gguf lives inside fakemodels/stray/node_modules/ — must be skipped.
        self.assertNotIn("hidden.gguf", paths)

    def test_entries_are_tagged_as_other_source(self) -> None:
        entries = list(StrayScanner(roots=[FIXTURES], skip_dirs=set()).scan())
        self.assertTrue(entries)
        for entry in entries:
            self.assertEqual(entry.source, SOURCE_OTHER)

    def test_skip_dirs_excludes_subtree_from_results(self) -> None:
        entries = list(
            StrayScanner(roots=[FIXTURES], skip_dirs={FIXTURES / "notes"}).scan()
        )
        names = {Path(e.path).name for e in entries}
        self.assertIn("checkpoint.pt", names)
        self.assertNotIn("notes-llm.gguf", names)
        self.assertNotIn("random.safetensors", names)

    def test_missing_root_is_silently_skipped(self) -> None:
        entries = list(
            StrayScanner(
                roots=[FIXTURES / "does-not-exist", FIXTURES],
                skip_dirs=set(),
            ).scan()
        )
        # The valid root still contributes despite the missing one.
        self.assertTrue(entries)

    def test_ignore_marker_excludes_directory_subtree(self) -> None:
        # A directory containing a .modelherderignore marker (and everything
        # below it) must be skipped entirely.
        ignored_dir = FIXTURES / "project" / "ignored_subtree"
        ignored_dir.mkdir(parents=True, exist_ok=True)
        marker = ignored_dir / ".modelherderignore"
        marker.write_text("")
        hidden = ignored_dir / "secret.gguf"
        hidden.write_bytes(b"x")
        try:
            entries = list(
                StrayScanner(roots=[FIXTURES], skip_dirs=set()).scan()
            )
            names = {Path(e.path).name for e in entries}
            self.assertNotIn("secret.gguf", names)
            # Sibling files outside the ignored subtree still show up.
            self.assertIn("checkpoint.pt", names)
        finally:
            hidden.unlink()
            marker.unlink()
            ignored_dir.rmdir()

    def test_bin_files_are_not_picked_up_by_stray_scan(self) -> None:
        # STRAY_EXTS intentionally excludes .bin — confirm by adding one and
        # making sure it is not in the results.
        bin_path = FIXTURES / "project" / "noise.bin"
        bin_path.write_bytes(b"x")
        try:
            entries = list(
                StrayScanner(roots=[FIXTURES], skip_dirs=set()).scan()
            )
            names = {Path(e.path).name for e in entries}
            self.assertNotIn("noise.bin", names)
        finally:
            bin_path.unlink()


if __name__ == "__main__":
    unittest.main()
