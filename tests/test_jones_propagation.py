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
from scattering_calculator.sample_generator.structures import CompactDielectricTensorStack


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

    def test_compact_dielectric_stack_matches_dense_stack(self) -> None:
        field = np.zeros((8, 9, 2), dtype=complex)
        field[..., 0] = 1.0
        field[..., 1] = 0.25j

        dense = np.zeros((2, 8, 9, 2, 2), dtype=complex)
        dense[0, ..., 0, 0] = 4.0
        dense[0, ..., 1, 1] = 5.0
        dense[1, ..., 0, 0] = 6.0
        dense[1, ..., 1, 1] = 7.0
        patch_region = (slice(2, 5), slice(3, 7))
        patch = dense[1, patch_region[0], patch_region[1]].copy()
        patch[..., 0, 1] = 0.1j
        patch[..., 1, 0] = -0.1j
        dense[1, patch_region[0], patch_region[1]] = patch

        compact = CompactDielectricTensorStack(
            shape=dense.shape,
            base_diagonal=np.array([[4.0, 5.0], [6.0, 7.0]], dtype=complex),
            patches=((), ((patch_region, patch),)),
            aperture_support_regions=(patch_region,),
        )

        wf = object.__new__(wavefronts)
        wf.aperture_support_regions = None
        dense_out = wf.propagate_jones_multislice(
            field,
            dense,
            wavelength=1e-9,
            thicknesses=[2e-9, 3e-9],
            pixel_size=1e-9,
            propagate=False,
        )
        compact_out = wf.propagate_jones_multislice(
            field,
            compact,
            wavelength=1e-9,
            thicknesses=[2e-9, 3e-9],
            pixel_size=1e-9,
            propagate=False,
        )

        self.assertTrue(np.allclose(compact_out, dense_out))


if __name__ == "__main__":
    unittest.main()
