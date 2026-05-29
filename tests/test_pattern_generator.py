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


if __name__ == "__main__":
    unittest.main()
