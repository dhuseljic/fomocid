"""Test magnetic pattern generator behavior and configuration wiring."""

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
from scattering_calculator.sample_generator.structures import Apertures3D
from scattering_calculator.simulation_pipelines import simulation_configuration


class BinaryLabyrinthAutoSizeTests(unittest.TestCase):
    def test_auto_size_shrinks_generated_field_for_large_target_stripes(self) -> None:
        """Test that auto size shrinks generated field for large target stripes.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        calls: list[tuple[int, int]] = []

        def fake_generate_binary(**kwargs):
            """Run the fake generate binary operation.

            Parameters
            ----------
            **kwargs : Any
                Input value for ``kwargs``.

            Returns
            -------
            result : Any
                Return value produced by the function.
            """
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
        """Test that auto size false respects requested field size.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        calls: list[tuple[int, int]] = []

        def fake_generate_binary(**kwargs):
            """Run the fake generate binary operation.

            Parameters
            ----------
            **kwargs : Any
                Input value for ``kwargs``.

            Returns
            -------
            result : Any
                Return value produced by the function.
            """
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


class ImagePatternTests(unittest.TestCase):
    def test_image_pattern_rescales_thresholded_domains(self) -> None:
        """Test that image pattern rescales thresholded domains.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        image = np.zeros((4, 4), dtype=float)
        image[:, 2:] = 1.0

        pattern, meta = pattern_generator.create_image_pattern(
            (8, 8),
            image_array=image,
            image_pixel_size=2.0,
            real_space_pixel_size=1.0,
            sigma=0.0,
            threshold=0.5,
        )

        self.assertEqual(pattern.shape, (8, 8))
        np.testing.assert_array_equal(pattern[:, :4], -1.0)
        np.testing.assert_array_equal(pattern[:, 4:], 1.0)
        self.assertEqual(meta["rescale_factor"], 2.0)
        self.assertEqual(meta["source"], "image_array")

    def test_image_pattern_sigma_is_converted_by_config(self) -> None:
        """Test that image pattern sigma is converted by config.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        image = np.zeros((6, 6), dtype=float)
        image[:, 3:] = 1.0

        config = simulation_configuration.MagneticPatternConfig(
            pattern_type_method="image_pattern",
            shape=(12, 12),
            real_space_pixel_size=1e-9,
            pattern_config={
                "image_array": image,
                "image_pixel_size": 2e-9,
                "sigma": 1e-9,
                "threshold": 0.5,
            },
        )
        pattern, meta = config.create_pattern()

        self.assertEqual(pattern.shape, (12, 12))
        self.assertAlmostEqual(meta["sigma_px"], 1.0)
        self.assertAlmostEqual(meta["rescale_factor"], 2.0)
        self.assertLess(np.min(np.abs(pattern)), 1.0)
        self.assertLessEqual(np.max(np.abs(pattern)), 1.0)


class DisorderedSkyrmionTests(unittest.TestCase):
    def test_small_skyrmions_are_sampled_as_continuous_profiles(self) -> None:
        """Test that subpixel skyrmions do not collapse to binary crosses."""
        pattern, coords = pattern_generator.create_disordered_skyrmion_lattice_pattern(
            (32, 32),
            stripe_width=2.4,
            sigma=0.0,
            skyrmion_density=0.15,
            diameter_spread=0.0,
            ellipticity=(1.0, 1.0),
            roughness=0.0,
            placement_center=(16.35, 15.65),
            placement_radius=3.0,
            seed=7,
        )

        self.assertGreaterEqual(len(coords), 1)
        self.assertTrue(np.any((pattern > -1.0) & (pattern < 1.0)))
        self.assertGreater(len(np.unique(np.round(pattern, 3))), 3)
        self.assertLess(np.min(pattern), -0.5)

    def test_subpixel_disordered_skyrmion_has_area_averaged_contrast(self) -> None:
        """Test that a subpixel skyrmion does not receive full contrast."""
        pattern, coords = pattern_generator.create_disordered_skyrmion_lattice_pattern(
            (17, 17),
            stripe_width=0.6,
            sigma=0.0,
            skyrmion_density=0.01,
            diameter_spread=0.0,
            ellipticity=(1.0, 1.0),
            roughness=0.0,
            placement_center=(8.0, 8.0),
            placement_radius=0.1,
            seed=7,
        )

        self.assertEqual(len(coords), 1)
        self.assertGreater(np.min(pattern), 0.0)
        self.assertLess(np.min(pattern), 1.0)

    def test_subpixel_circular_skyrmion_has_area_averaged_contrast(self) -> None:
        """Test that the legacy circular skyrmion generator preserves area fraction."""
        pattern, coords = pattern_generator.create_skyrmion_pattern(
            [17, 17],
            skyr_radius=0.3,
            screening_radius=1.0,
            number_skyr=1,
            number_iter=1,
            sigma=0.0,
            seed=1,
        )

        self.assertEqual(len(coords), 1)
        self.assertGreater(np.min(pattern), 0.0)
        self.assertLess(np.min(pattern), 1.0)


class ApertureAreaAverageTests(unittest.TestCase):
    def test_subpixel_aperture_hole_has_area_averaged_transmission(self) -> None:
        """Test that a subpixel aperture hole is partial, not binary."""
        aperture = Apertures3D(
            shape=(1, 17, 17),
            real_space_pixel_size=1.0,
            layer_thicknesses=[1.0],
        )
        aperture.create_circle_aperture(
            center=(8.0, 8.0),
            radius=0.3,
            depth=1,
            use_real_space_coordinates=False,
            sigma=0.0,
            top_radius_factor=1.0,
            use_roi=True,
        )

        transmission = aperture.return_aperture_mask()[0]
        self.assertGreater(np.min(transmission), 0.0)
        self.assertLess(np.min(transmission), 1.0)
        self.assertAlmostEqual(1.0 - np.min(transmission), np.pi * 0.3**2, places=2)


if __name__ == "__main__":
    unittest.main()
