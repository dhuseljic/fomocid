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
    DynamicProjectedDielectricTensorStack,
    Structure,
)


class CompactDielectricTensorTests(unittest.TestCase):
    def _projected_structure(self) -> Structure:
        structure = object.__new__(Structure)
        structure.mask = np.ones((1, 2, 3), dtype=float)
        structure.magnetization = np.zeros((1, 2, 3, 3), dtype=float)
        structure.layer_refractive_indices = np.array(
            [[2.0 + 0.0j, 0.1 + 0.0j, 0.0j]],
            dtype=complex,
        )
        structure.layer_thicknesses = [5e-9]
        structure.sample_shape = [1, 2, 3]
        structure.real_space_pixel_size = 1e-9
        structure.sample_tilt_theta = 0.0
        structure.sample_tilt_axis = "x"
        structure.sample_tilt_voxel_size = None
        structure.sample_tilt_antialias_samples = 1
        structure.sample_tilt_simulation_z_extent = None
        return structure

    def test_beam_direction_projection_matches_normal_xmcd_limit(self) -> None:
        """Projected Jones tensor preserves the normal-incidence circular term."""
        structure = self._projected_structure()
        structure.magnetization[..., 2] = 1.0

        structure.calculate_final_dielectric_tensor(
            compact=False,
            beam_direction=(0.0, 0.0, 1.0),
        )

        eps = structure.final_dielectric_tensor
        np.testing.assert_allclose(eps[..., 0, 0], 4.0)
        np.testing.assert_allclose(eps[..., 1, 1], 4.0)
        np.testing.assert_allclose(eps[..., 0, 1], 0.4j)
        np.testing.assert_allclose(eps[..., 1, 0], -0.4j)

    def test_beam_direction_projection_uses_magnetization_along_beam(self) -> None:
        """XMCD follows m dot k, not always the lab z component."""
        structure = self._projected_structure()
        structure.magnetization[..., 0] = 1.0

        structure.calculate_final_dielectric_tensor(
            compact=False,
            beam_direction=(1.0, 0.0, 0.0),
        )

        eps = structure.final_dielectric_tensor
        np.testing.assert_allclose(eps[..., 0, 0], 4.0)
        np.testing.assert_allclose(eps[..., 1, 1], 4.0)
        np.testing.assert_allclose(eps[..., 0, 1], 0.4j)
        np.testing.assert_allclose(eps[..., 1, 0], -0.4j)

    def test_beam_direction_projection_rejects_zero_direction(self) -> None:
        """A zero beam direction is physically undefined."""
        structure = self._projected_structure()

        with self.assertRaisesRegex(ValueError, "beam_direction"):
            structure.calculate_final_dielectric_tensor(
                compact=False,
                beam_direction=(0.0, 0.0, 0.0),
            )

    def test_local_k_projection_stack_projects_xmcd_from_k_map(self) -> None:
        """Dynamic stack uses the supplied local k map for m dot k."""
        structure = self._projected_structure()
        structure.magnetization[..., 0] = 1.0

        structure.calculate_final_dielectric_tensor(
            compact=False,
            local_k_projection=True,
        )
        stack = structure.final_dielectric_tensor
        self.assertIsInstance(stack, DynamicProjectedDielectricTensorStack)

        k_map = np.zeros((2, 3, 3), dtype=float)
        k_map[..., 0] = 0.5
        k_map[..., 2] = np.sqrt(1.0 - 0.5**2)
        eps = stack.project_layer(0, k_map)

        np.testing.assert_allclose(eps[..., 0, 0], 4.0)
        np.testing.assert_allclose(eps[..., 1, 1], 4.0)
        np.testing.assert_allclose(eps[..., 0, 1], 0.2j)
        np.testing.assert_allclose(eps[..., 1, 0], -0.2j)

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
