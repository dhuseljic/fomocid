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


if __name__ == "__main__":
    unittest.main()
