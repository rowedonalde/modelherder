from __future__ import annotations

import unittest

from modelherder.sizes import human_size


KiB = 1024
MiB = KiB * 1024
GiB = MiB * 1024
TiB = GiB * 1024
PiB = TiB * 1024


class HumanSizeTests(unittest.TestCase):
    def test_negative_returns_question_mark(self) -> None:
        for n in (-1, -1024, -(10**12)):
            with self.subTest(n=n):
                self.assertEqual(human_size(n), "?")

    def test_zero_bytes(self) -> None:
        self.assertEqual(human_size(0), "0 B")

    def test_sub_kb_integer_formatting(self) -> None:
        # Bytes unit is the only one without a decimal place.
        for n, expected in [(1, "1 B"), (512, "512 B"), (1023, "1023 B")]:
            with self.subTest(n=n):
                self.assertEqual(human_size(n), expected)

    def test_exact_kb_boundary(self) -> None:
        # 1024 flips out of the bytes branch into KiB with .1f formatting.
        self.assertEqual(human_size(KiB), "1.0 KiB")

    def test_mid_unit_values(self) -> None:
        cases = [
            (1536, "1.5 KiB"),
            (int(1.5 * MiB), "1.5 MiB"),
            (int(2.5 * GiB), "2.5 GiB"),
        ]
        for n, expected in cases:
            with self.subTest(n=n):
                self.assertEqual(human_size(n), expected)

    def test_each_unit_boundary(self) -> None:
        cases = [
            (KiB, "1.0 KiB"),
            (MiB, "1.0 MiB"),
            (GiB, "1.0 GiB"),
            (TiB, "1.0 TiB"),
            (PiB, "1.0 PiB"),
        ]
        for n, expected in cases:
            with self.subTest(unit=expected):
                self.assertEqual(human_size(n), expected)

    def test_above_pb_clamps_to_pb(self) -> None:
        # 1024 ** 6 has no larger unit defined; it should still render as PiB.
        result = human_size(1024**6)
        self.assertTrue(result.endswith(" PiB"), result)
        self.assertEqual(result, "1024.0 PiB")

    def test_rounding_one_decimal(self) -> None:
        # 1124 / 1024 ≈ 1.0976... → rounded to 1.1 KiB by .1f
        self.assertEqual(human_size(1124), "1.1 KiB")
