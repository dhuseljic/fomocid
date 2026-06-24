"""Test multilayer recipe parsing and effective-medium layers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scattering_calculator.database.database_loading import material_params
from scattering_calculator.sample_generator.structures import Structure, parse_recipe


class RecipeParserTests(unittest.TestCase):
    def test_adjacent_material_terms_create_composite_layer(self) -> None:
        """Test that adjacent material terms create composite layer.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        recipe = parse_recipe("Pt(4)Co(6)/SiN(20)")

        self.assertEqual(len(recipe.layers), 2)
        self.assertTrue(recipe.layers[0].is_composite)
        self.assertEqual(recipe.layers[0].material, "Pt(4)Co(6)")
        self.assertAlmostEqual(recipe.layers[0].thickness, 10e-9)
        self.assertEqual(
            [component[0] for component in recipe.layers[0].components],
            ["Pt", "Co"],
        )
        np.testing.assert_allclose(
            [component[1] for component in recipe.layers[0].components],
            [4e-9, 6e-9],
        )
        self.assertFalse(recipe.layers[1].is_composite)

    def test_slash_separated_material_terms_remain_separate_layers(self) -> None:
        """Test that slash separated material terms remain separate layers.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        recipe = parse_recipe("Pt(4)/Co(6)")

        self.assertEqual(len(recipe.layers), 2)
        self.assertFalse(recipe.layers[0].is_composite)
        self.assertFalse(recipe.layers[1].is_composite)
        self.assertEqual([layer.material for layer in recipe.layers], ["Pt", "Co"])

    def test_composite_layer_accepts_any_number_of_material_terms(self) -> None:
        """Test that composite layer accepts any number of material terms.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        recipe = parse_recipe("Pt(1)Co(4)Ta(3)Au(2)")
        layer = recipe.layers[0]

        self.assertEqual(len(recipe.layers), 1)
        self.assertTrue(layer.is_composite)
        self.assertEqual(layer.material, "Pt(1)Co(4)Ta(3)Au(2)")
        self.assertAlmostEqual(layer.thickness, 10e-9)
        self.assertEqual(
            [component[0] for component in layer.components],
            ["Pt", "Co", "Ta", "Au"],
        )
        np.testing.assert_allclose(
            [component[1] for component in layer.components],
            [1e-9, 4e-9, 3e-9, 2e-9],
        )

    def test_composite_layer_uses_thickness_weighted_dielectric_tensor(self) -> None:
        """Test that composite layer uses thickness weighted dielectric tensor.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        params = material_params(
            refractive_indices={
                "Pt": np.array([2.0 + 0.0j, 0.2 + 0.0j, 0.02 + 0.0j]),
                "Co": np.array([4.0 + 0.0j, 0.4 + 0.0j, 0.04 + 0.0j]),
            }
        )
        structure = Structure(
            name="test",
            material_params=params,
            sample_shape=[0, 4, 4],
            real_space_pixel_size=1e-9,
        )

        structure.add_effective_layer(
            "Pt(4)Co(6)",
            (("Pt", 4e-9), ("Co", 6e-9)),
        )

        pt_tensor = structure.dielectric_tensor_mixed(
            params.get_refractive_index("Pt")
        )
        co_tensor = structure.dielectric_tensor_mixed(
            params.get_refractive_index("Co")
        )
        expected = 0.4 * pt_tensor + 0.6 * co_tensor

        self.assertEqual(structure.layer_names, ["Pt(4)Co(6)"])
        self.assertAlmostEqual(structure.layer_thicknesses[0], 10e-9)
        np.testing.assert_allclose(structure.dielectric_tensors[0], expected)

    def test_tilted_layer_voxelization_adds_vacuum_and_fractional_interfaces(self) -> None:
        """Test tilted sample voxelization with interpolated material fractions."""
        params = material_params(
            refractive_indices={
                "Co": np.array([2.0 + 0.0j, 0.0j, 0.0j]),
            }
        )
        structure = Structure(
            name="tilted",
            material_params=params,
            sample_shape=[0, 5, 9],
            real_space_pixel_size=1.0,
        )
        structure.add_layer("Co", thickness=2.0)
        structure.mask = np.ones((1, 5, 9), dtype=float)
        structure.magnetization = np.zeros((1, 5, 9, 3), dtype=float)
        structure.set_sample_tilt(
            theta=np.deg2rad(30.0),
            axis="x",
            voxel_size=0.5,
            antialias_samples=3,
        )

        structure.calculate_final_dielectric_tensor(compact=False)
        eps = structure.final_dielectric_tensor

        self.assertGreater(eps.shape[0], 4)
        self.assertEqual(len(structure.propagation_layer_thicknesses), eps.shape[0])
        material_fraction = (eps[..., 0, 0].real - 1.0) / (2.0**2 - 1.0)
        self.assertLess(np.min(material_fraction), 1e-12)
        self.assertGreater(np.max(material_fraction), 1.0 - 1e-12)
        self.assertTrue(np.any((material_fraction > 0.0) & (material_fraction < 1.0)))

    def test_zero_tilt_keeps_original_compact_path(self) -> None:
        """Test theta=0 does not expand the layer stack."""
        params = material_params(
            refractive_indices={
                "Co": np.array([2.0 + 0.0j, 0.0j, 0.0j]),
            }
        )
        structure = Structure(
            name="untilted",
            material_params=params,
            sample_shape=[0, 3, 3],
            real_space_pixel_size=1.0,
        )
        structure.add_layer("Co", thickness=2.0)
        structure.mask = np.ones((1, 3, 3), dtype=float)
        structure.magnetization = np.zeros((1, 3, 3, 3), dtype=float)
        structure.set_sample_tilt(theta=0.0)

        structure.calculate_final_dielectric_tensor(compact=True)

        self.assertEqual(structure.final_dielectric_tensor.shape[0], 1)
        self.assertEqual(structure.propagation_layer_thicknesses, [2.0])

    def test_zero_tilt_fixed_volume_uses_tilted_magnetization(self) -> None:
        """Test theta=0 still supports a fixed simulation z volume."""
        params = material_params(
            refractive_indices={
                "Co": np.array([2.0 + 0.0j, 0.5 + 0.0j, 0.0j]),
            }
        )
        structure = Structure(
            name="untilted-fixed-volume",
            material_params=params,
            sample_shape=[0, 3, 4],
            real_space_pixel_size=1.0,
        )
        structure.add_layer("Co", thickness=2.0)
        structure.mask = np.ones((1, 3, 4), dtype=float)
        structure.magnetization = np.zeros((1, 3, 4, 3), dtype=float)
        structure.set_sample_tilt(
            theta=0.0,
            voxel_size=1.0,
            simulation_z_extent=6.0,
        )
        _, film_x, film_depth = structure.tilted_material_coordinate_grids()
        structure.tilted_magnetization = np.zeros((*film_x.shape, 3), dtype=float)
        structure.tilted_magnetization[..., 2] = np.sign(film_x)

        structure.calculate_final_scalar_refractive_index("CR", compact=False)
        n_stack = structure.final_scalar_refractive_index

        self.assertEqual(n_stack.shape, (6, 3, 4))
        self.assertEqual(len(structure.propagation_layer_thicknesses), 6)
        self.assertTrue(np.any(np.isclose(n_stack.real, 1.0)))
        material_slice = np.argmin(np.abs(film_depth[:, 0, 0] - 1.0))
        self.assertLess(np.min(n_stack[material_slice].real), 2.0)
        self.assertGreater(np.max(n_stack[material_slice].real), 2.0)

    def test_tilted_fixed_volume_center_offset_moves_slab(self) -> None:
        """Test lab-frame centre offsets move the voxelized slab."""
        params = material_params(
            refractive_indices={
                "Co": np.array([2.0 + 0.0j, 0.0j, 0.0j]),
            }
        )
        structure = Structure(
            name="offset-fixed-volume",
            material_params=params,
            sample_shape=[0, 3, 5],
            real_space_pixel_size=1.0,
        )
        structure.add_layer("Co", thickness=2.0)
        structure.mask = np.ones((1, 3, 5), dtype=float)
        structure.magnetization = np.zeros((1, 3, 5, 3), dtype=float)
        structure.set_sample_tilt(
            theta=0.0,
            voxel_size=1.0,
            simulation_z_extent=6.0,
            center_offset=(0.5, 1.5, 1.0),
        )

        film_y, film_x, film_depth = structure.tilted_material_coordinate_grids()
        fractions, thicknesses = structure._tilted_layer_fractions()

        self.assertEqual(thicknesses, [1.0] * 6)
        np.testing.assert_allclose(film_y[:, 1, 2], -0.5)
        np.testing.assert_allclose(film_x[:, 1, 2], -1.5)

        z_centers = (np.arange(6, dtype=float) - 6 / 2 + 0.5)
        occupied = fractions[:, 1, 2, 0] > 0.5
        np.testing.assert_array_equal(z_centers[occupied], np.array([0.5, 1.5]))
        np.testing.assert_allclose(film_depth[:, 1, 2], z_centers)

    def test_ninety_degree_tilt_requires_and_uses_fixed_volume(self) -> None:
        """Test that 90 degree tilt works with an explicit simulation volume."""
        params = material_params(
            refractive_indices={
                "Co": np.array([2.0 + 0.0j, 0.0j, 0.0j]),
            }
        )
        structure = Structure(
            name="vertical-film",
            material_params=params,
            sample_shape=[0, 3, 7],
            real_space_pixel_size=1.0,
        )
        structure.add_layer("Co", thickness=3.0)
        structure.mask = np.ones((1, 3, 7), dtype=float)
        structure.magnetization = np.zeros((1, 3, 7, 3), dtype=float)
        structure.set_sample_tilt(
            theta=np.pi / 2,
            axis="x",
            voxel_size=1.0,
            simulation_z_extent=5.0,
        )

        structure.calculate_final_scalar_refractive_index("CR", compact=False)
        n_stack = structure.final_scalar_refractive_index

        self.assertEqual(n_stack.shape, (5, 3, 7))
        self.assertTrue(np.any(np.isclose(n_stack.real, 2.0)))
        self.assertTrue(np.any(np.isclose(n_stack.real, 1.0)))


if __name__ == "__main__":
    unittest.main()
