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
from scattering_calculator.sample_generator.domain_analysis import (
    classify_magnetic_domains,
)
from scattering_calculator.sample_generator.gray_scott_generator_binary import (
    generate as generate_binary,
)
from scattering_calculator.sample_generator.structures import Apertures3D
from scattering_calculator.simulation_pipelines import simulation_configuration


class BinaryDomainPhaseSpaceTests(unittest.TestCase):
    def test_target_mean_changes_raw_binary_phase_fraction(self) -> None:
        """The evolution constraint must alter occupancy before thresholding."""
        fractions = []
        for target_mean in (-0.4, 0.0, 0.4):
            _, binary, continuous, _ = generate_binary(
                batch=1,
                H=64,
                W=64,
                n_steps=40,
                region="custom",
                use_gpu=False,
                seed=4,
                k0=0.7,
                eps=0.8,
                noise_amp=0.0,
                target_mean=target_mean,
            )
            self.assertAlmostEqual(float(continuous.mean()), target_mean, places=5)
            fractions.append(float(np.mean(binary < 0)))

        self.assertGreater(fractions[0], fractions[1])
        self.assertGreater(fractions[1], fractions[2])

    def test_spatial_field_bias_suppresses_one_phase_outside_roi(self) -> None:
        """A spatial evolution field can confine minority domains."""
        height = width = 64
        yy, xx = np.indices((height, width))
        inside = (yy - height / 2) ** 2 + (xx - width / 2) ** 2 < 18**2
        field_bias = np.where(inside, 0.0, 0.08)

        _, binary, _, _ = generate_binary(
            batch=1,
            H=height,
            W=width,
            n_steps=60,
            region="custom",
            use_gpu=False,
            seed=2,
            k0=0.7,
            eps=0.8,
            noise_amp=0.0,
            field_bias=field_bias,
        )

        self.assertGreater(np.mean(binary[0, inside] < 0), 0.2)
        self.assertLess(np.mean(binary[0, ~inside] < 0), 0.1)

    def test_field_bias_rejects_incompatible_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "field_bias must be"):
            generate_binary(
                batch=1,
                H=32,
                W=32,
                n_steps=1,
                use_gpu=False,
                field_bias=np.zeros((12, 12)),
            )


class MagneticDomainClassificationTests(unittest.TestCase):
    def test_counts_round_minority_components_as_bubbles(self) -> None:
        yy, xx = np.indices((96, 96))
        pattern = np.ones((96, 96), dtype=float)
        for center_y, center_x, radius in ((25, 25, 8), (68, 62, 10)):
            circle = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
            pattern[circle] = -1.0

        analysis = classify_magnetic_domains(pattern)

        self.assertEqual(analysis["minority_polarity"], -1)
        self.assertEqual(analysis["bubble_count"], 2)
        self.assertEqual(analysis["stripe_count"], 0)
        self.assertEqual(analysis["morphology"], "bubbles")

    def test_classifies_elongated_domain_as_stripe(self) -> None:
        pattern = np.ones((64, 64), dtype=float)
        pattern[28:36, 8:56] = -1.0

        analysis = classify_magnetic_domains(pattern)

        self.assertEqual(analysis["bubble_count"], 0)
        self.assertEqual(analysis["stripe_count"], 1)
        self.assertEqual(analysis["morphology"], "stripes")
        self.assertGreater(
            analysis["minority_components"][0]["eccentricity"], 0.8
        )

    def test_boundary_clipped_circle_is_not_counted_as_bubble(self) -> None:
        yy, xx = np.indices((64, 64))
        pattern = np.ones((64, 64), dtype=float)
        pattern[(yy - 30) ** 2 + xx**2 <= 10**2] = -1.0

        analysis = classify_magnetic_domains(pattern)

        self.assertEqual(analysis["bubble_count"], 0)
        self.assertEqual(analysis["stripe_count"], 1)

    def test_binarizes_continuous_input_at_requested_threshold(self) -> None:
        pattern = np.ones((32, 32), dtype=float)
        pattern[10:22, 10:22] = -0.2

        analysis = classify_magnetic_domains(pattern, threshold=0.5)

        self.assertEqual(set(np.unique(analysis["binary_pattern"])), {-1, 1})
        self.assertEqual(analysis["bubble_count"], 1)

    def test_analysis_mask_limits_counts_to_object_hole(self) -> None:
        yy, xx = np.indices((80, 80))
        field_of_view = (yy - 40) ** 2 + (xx - 40) ** 2 <= 25**2
        pattern = np.ones((80, 80), dtype=float)
        pattern[(yy - 40) ** 2 + (xx - 40) ** 2 <= 7**2] = -1.0
        pattern[(yy - 8) ** 2 + (xx - 8) ** 2 <= 5**2] = -1.0

        analysis = classify_magnetic_domains(
            pattern, analysis_mask=field_of_view
        )

        self.assertEqual(analysis["bubble_count"], 1)

    def test_small_nested_sign_hole_is_filled(self) -> None:
        field = -np.ones((40, 40), dtype=float)
        field[8:32, 8:32] = 1.0
        field[19:21, 19:21] = -1.0

        cleaned = pattern_generator.fill_small_domain_holes(field, max_hole_area=4)

        self.assertTrue(np.all(cleaned[19:21, 19:21] > 0))


class BinaryLabyrinthAutoSizeTests(unittest.TestCase):
    def test_soft_conversion_preserves_saturated_target_mean_sign(self) -> None:
        with mock.patch.object(
            pattern_generator,
            "_estimate_labyrinth_stripe_width_fft",
            side_effect=AssertionError("saturated state must skip FFT sizing"),
        ):
            negative, negative_meta = pattern_generator.create_binary_labyrinth_pattern(
                (32, 32),
                stripe_width=4,
                sigma=1,
                H=96,
                W=96,
                n_steps=30,
                region="custom",
                use_gpu=False,
                seed=3,
                k0=1.1,
                eps=1.2,
                target_mean=-1.0,
            )
        positive, _ = pattern_generator.create_binary_labyrinth_pattern(
            (32, 32),
            stripe_width=4,
            sigma=1,
            H=32,
            W=32,
            n_steps=10,
            region="custom",
            use_gpu=False,
            seed=3,
            k0=1.1,
            eps=1.2,
            target_mean=1.0,
        )

        np.testing.assert_array_equal(negative, -1.0)
        self.assertTrue(negative_meta["saturated_shortcut"])
        self.assertEqual(negative_meta["rescale_factor"], 1.0)
        self.assertGreater(float(np.mean(positive)), 0.0)
        self.assertGreater(float(np.mean(positive > 0)), 0.8)

    def test_auto_size_rejects_pathological_fft_resize(self) -> None:
        """An anomalous measured width must fail before a huge second FFT."""
        calls: list[tuple[int, int]] = []

        def fake_generate_binary(**kwargs):
            height = int(kwargs["H"])
            width = int(kwargs["W"])
            calls.append((height, width))
            # A nearly constant field makes the mocked width estimate below
            # the only driver of the adaptive resize.
            continuous = np.zeros((1, height, width), dtype=float)
            return None, None, continuous, {"H": height, "W": width}

        with (
            mock.patch.object(
                pattern_generator, "generate_binary", fake_generate_binary
            ),
            mock.patch.object(
                pattern_generator,
                "_estimate_labyrinth_stripe_width_fft",
                return_value=(1_000_000.0, 2_000_000.0),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "unsafe source field"):
                pattern_generator.create_binary_labyrinth_pattern(
                    (300, 300),
                    stripe_width=10,
                    n_steps=1,
                    auto_size=True,
                    max_auto_size=512,
                    max_auto_pixels=512**2,
                )

        self.assertEqual(len(calls), 1)

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

    def test_slit_aperture_uses_width_length_and_tapers(self) -> None:
        """Test rectangular slit aperture generation with a tapered top."""
        aperture = Apertures3D(
            shape=(3, 41, 41),
            real_space_pixel_size=1.0,
            layer_thicknesses=[1.0, 1.0, 1.0],
        )
        aperture.create_slit_aperture(
            center=(20.0, 20.0),
            width=4.0,
            length=14.0,
            depth=3,
            use_real_space_coordinates=False,
            sigma=0.0,
            angle=np.deg2rad(20),
            top_radius_factor=2.0,
            taper_depth=3.0,
            use_roi=True,
        )

        hole_fraction = 1.0 - aperture.return_aperture_mask()
        self.assertGreater(np.sum(hole_fraction[0]), np.sum(hole_fraction[-1]))
        self.assertGreater(np.sum(hole_fraction[-1]), 4.0 * 14.0 * 0.7)
        self.assertLess(np.min(aperture.return_aperture_mask()), 1.0)

    def test_front_aperture_accepts_per_aperture_depth(self) -> None:
        """Test that apertures can stop at independent depths."""
        config = simulation_configuration.FrontApertureConfig(
            aperture_shape=(3, 41, 41),
            real_space_pixel_size=1.0,
            aperture_thicknesses=[1.0, 1.0, 1.0],
            aperture_layer_names=["Au", "SiN", "Co"],
            aperture_config={
                "apertures_type": ["OH", "RH"],
                "apertures_radius": [4.0, 3.0],
                "apertures_center": [(0.0, 0.0), (10.0, 0.0)],
                "apertures_sigma": [0.0, 0.0],
                "apertures_depth": [1.0, 2.0],
                "apertures_top_radius_factor": [1.0, 1.0],
            },
        )
        config.setup()
        mask = config.return_aperture()

        center = (20, 20)
        reference = (30, 20)
        self.assertLess(mask[0, center[0], center[1]], 1.0)
        self.assertEqual(mask[1, center[0], center[1]], 1.0)
        self.assertLess(mask[0, reference[0], reference[1]], 1.0)
        self.assertLess(mask[1, reference[0], reference[1]], 1.0)
        self.assertEqual(mask[2, reference[0], reference[1]], 1.0)

    def test_object_hole_defaults_to_before_silicon_nitride(self) -> None:
        """Test legacy OH depth default still stops before the SiN layer."""
        config = simulation_configuration.FrontApertureConfig(
            aperture_shape=(3, 31, 31),
            real_space_pixel_size=1.0,
            aperture_thicknesses=[1.0, 1.0, 1.0],
            aperture_layer_names=["Au", "SiN", "Co"],
            aperture_config={
                "apertures_type": ["OH"],
                "apertures_radius": [4.0],
                "apertures_center": [(0.0, 0.0)],
                "apertures_sigma": [0.0],
                "thickness_OH": 1.0,
                "apertures_top_radius_factor": [1.0],
            },
        )
        config.setup()
        mask = config.return_aperture()

        self.assertLess(mask[0, 15, 15], 1.0)
        self.assertEqual(mask[1, 15, 15], 1.0)


if __name__ == "__main__":
    unittest.main()
