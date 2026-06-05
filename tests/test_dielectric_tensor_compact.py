"""Test compact dielectric tensor construction against dense tensors."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scattering_calculator.sample_generator.structures import (
    CompactDielectricTensorStack,
    Structure,
)


class CompactDielectricTensorTests(unittest.TestCase):
    def test_compact_builder_materializes_like_dense_builder(self) -> None:
        """Test that compact builder materializes like dense builder.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        structure = object.__new__(Structure)
        structure.mask = np.ones((2, 6, 7), dtype=float)
        structure.mask[0, 2:4, 3:5] = 0.0
        structure.magnetization = np.zeros((2, 6, 7, 3), dtype=float)
        structure.magnetization[1, 2:4, 3:5, 2] = 1.0
        structure.dielectric_tensors = np.zeros((2, 3, 2, 2), dtype=complex)
        structure.dielectric_tensors[0, 0, 0, 0] = 4.0
        structure.dielectric_tensors[0, 0, 1, 1] = 5.0
        structure.dielectric_tensors[1, 0, 0, 0] = 6.0
        structure.dielectric_tensors[1, 0, 1, 1] = 7.0
        structure.dielectric_tensors[1, 1, 0, 1] = 0.2j
        structure.dielectric_tensors[1, 1, 1, 0] = -0.2j

        compact = structure.calculate_compact_dielectric_tensor(use_aperture_roi=True)
        structure.calculate_final_dielectric_tensor(
            use_aperture_roi=True,
            compact=False,
        )

        self.assertIsInstance(compact, CompactDielectricTensorStack)
        self.assertEqual(compact.shape, structure.final_dielectric_tensor.shape)
        self.assertTrue(np.allclose(compact.materialize(), structure.final_dielectric_tensor))
        self.assertEqual(len(compact.patches[0]), 1)
        self.assertEqual(len(compact.patches[1]), 1)

    def test_overlapping_aperture_support_is_one_physical_roi(self) -> None:
        """Test that intersecting aperture supports cannot become separate ROIs."""
        structure = object.__new__(Structure)
        structure.mask = np.ones((2, 10, 10), dtype=float)
        structure.mask[0, 2:6, 2:6] = 0.0
        structure.mask[1, 4:8, 4:8] = 0.0
        structure.magnetization = np.zeros((2, 10, 10, 3), dtype=float)
        structure.dielectric_tensors = np.zeros((2, 3, 2, 2), dtype=complex)
        structure.dielectric_tensors[:, 0, 0, 0] = 4.0
        structure.dielectric_tensors[:, 0, 1, 1] = 5.0

        compact = structure.calculate_compact_dielectric_tensor(use_aperture_roi=True)

        self.assertIsNotNone(compact.aperture_support_regions)
        self.assertEqual(len(compact.aperture_support_regions), 1)
        y_slice, x_slice = compact.aperture_support_regions[0]
        self.assertEqual((y_slice.start, y_slice.stop), (2, 8))
        self.assertEqual((x_slice.start, x_slice.stop), (2, 8))


if __name__ == "__main__":
    unittest.main()
