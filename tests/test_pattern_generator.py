from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scattering_calculator.sample_generator import pattern_generator


class BinaryLabyrinthAutoSizeTests(unittest.TestCase):
    def test_auto_size_shrinks_generated_field_for_large_target_stripes(self) -> None:
        calls: list[tuple[int, int]] = []

        def fake_generate_binary(**kwargs):
            height = int(kwargs["H"])
            width = int(kwargs["W"])
            calls.append((height, width))
            x = np.arange(width, dtype=float)[None, :]
            continuous = np.sin(2.0 * np.pi * x / 8.0)
            continuous = np.broadcast_to(continuous, (height, width))
            return None, None, continuous[np.newaxis, :, :], {"H": height, "W": width}

        with mock.patch.object(pattern_generator, "generate_binary", fake_generate_binary):
            pattern, meta = pattern_generator.create_binary_labyrinth_pattern(
                (300, 300),
                stripe_width=100,
                H=100,
                W=100,
                n_steps=1,
                region="custom",
                k0=np.pi / 4.0,
                auto_size=True,
            )

        self.assertEqual(pattern.shape, (300, 300))
        self.assertLess(calls[0][0], 100)
        self.assertLess(calls[0][1], 100)
        self.assertEqual(meta["generated_H"], calls[-1][0])
        self.assertEqual(meta["generated_W"], calls[-1][1])
        self.assertGreaterEqual(meta["scaled_H"], 300)
        self.assertGreaterEqual(meta["scaled_W"], 300)

    def test_auto_size_false_respects_requested_field_size(self) -> None:
        calls: list[tuple[int, int]] = []

        def fake_generate_binary(**kwargs):
            height = int(kwargs["H"])
            width = int(kwargs["W"])
            calls.append((height, width))
            x = np.arange(width, dtype=float)[None, :]
            continuous = np.sin(2.0 * np.pi * x / 8.0)
            continuous = np.broadcast_to(continuous, (height, width))
            return None, None, continuous[np.newaxis, :, :], {"H": height, "W": width}

        with mock.patch.object(pattern_generator, "generate_binary", fake_generate_binary):
            pattern_generator.create_binary_labyrinth_pattern(
                (300, 300),
                stripe_width=100,
                H=100,
                W=100,
                n_steps=1,
                region="custom",
                k0=np.pi / 4.0,
                auto_size=False,
            )

        self.assertEqual(calls, [(100, 100)])


if __name__ == "__main__":
    unittest.main()
