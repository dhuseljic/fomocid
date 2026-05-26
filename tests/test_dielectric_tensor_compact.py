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


if __name__ == "__main__":
    unittest.main()
