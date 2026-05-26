from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scattering_calculator.beam_propagator.Jones_propagator import wavefronts


class JonesFreeSpacePropagationTests(unittest.TestCase):
    def test_padded_propagation_preserves_original_shape(self) -> None:
        field = np.zeros((16, 18, 2), dtype=complex)
        field[8, 9, 0] = 1.0

        out = wavefronts.propagate_free_space_jones(
            wavefronts,
            field,
            wavelength=1e-9,
            dz=10e-9,
            pixel_size=1e-9,
            padding_px=4,
            padding_mode="edge",
            absorber_width_px=2,
            absorber_strength=4.0,
            absorber_profile="cosine",
        )

        self.assertEqual(out.shape, field.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_edge_absorber_is_one_in_the_center(self) -> None:
        absorber = wavefronts._edge_absorber(
            32,
            32,
            width_px=8,
            strength=6.0,
            profile="cosine",
        )

        self.assertLess(absorber[0, 0], 1.0)
        self.assertAlmostEqual(absorber[16, 16], 1.0)

    def test_absorber_profile_varies_smoothly(self) -> None:
        absorber = wavefronts._edge_absorber(
            32,
            32,
            width_px=8,
            strength=6.0,
            profile="smoothstep",
        )
        edge_to_center = absorber[:9, 16]

        self.assertTrue(np.all(np.diff(edge_to_center) > 0.0))
        self.assertAlmostEqual(edge_to_center[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
