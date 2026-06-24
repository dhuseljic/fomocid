"""Test detector hologram beamstop and threshold variants."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scattering_calculator.experimental_conditions.detector import (
    detector_hologram,
    detector_layout,
)
from scattering_calculator.simulation_pipelines.simulation_configuration import HologramConfig


class DetectorBeamstopTests(unittest.TestCase):
    def test_legacy_artifact_counts_per_photon_moves_to_detector_params(self) -> None:
        artifacts, detector_params = detector_hologram._normalize_counts_per_photon(
            {"counts_per_photon": 125, "sigma_photon": 0.5},
            {"noise_rms": 2},
        )

        self.assertNotIn("counts_per_photon", artifacts)
        self.assertEqual(detector_params["counts_per_photon"], 125)

    def test_conflicting_counts_per_photon_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Conflicting counts_per_photon"):
            detector_hologram._normalize_counts_per_photon(
                {"counts_per_photon": 100},
                {"counts_per_photon": 180},
            )

    def test_legacy_noise_alias_moves_to_canonical_name(self) -> None:
        params = detector_hologram._normalize_detector_param_aliases(
            {"noise_rms": 2}
        )

        self.assertNotIn("noise_rms", params)
        self.assertEqual(params["readout_noise_sigma"], 2)

    def test_conflicting_noise_alias_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Conflicting detector noise"):
            detector_hologram._normalize_detector_param_aliases(
                {"noise_rms": 2, "readout_noise_sigma": 3}
            )

    def _hologram(self) -> detector_hologram:
        """Handle the internal hologram operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : detector_hologram
            Return value produced by the function.
        """
        holo = object.__new__(detector_hologram)
        holo.hologram_detector = np.ones((5, 5), dtype=float) * 100.0
        holo.beamstop = SimpleNamespace(beamstop=np.zeros((5, 5), dtype=float))
        holo.beamstop.beamstop[2, 2] = 1.0
        holo.beam_parameters = SimpleNamespace(coherence_length=None)
        holo.coherence_length = None
        holo.exposure_time = 1.0
        holo.quantum_efficiency = 1.0
        holo.max_counts_per_image = None
        holo.number_frames = 1
        holo.counts_per_photon = 1.0
        holo.sigma_photon = 0.0
        holo.readout_noise_average = 0.0
        holo.readout_noise_sigma = 0.0
        holo.detector_threshold = 1e9
        holo.noise_seed = 7
        return holo

    def test_detected_hologram_can_skip_beamstop_mask(self) -> None:
        """Test that detected hologram can skip beamstop mask.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        masked = self._hologram()
        masked.add_noise(apply_beamstop_mask=True)

        unmasked = self._hologram()
        unmasked.add_noise(apply_beamstop_mask=False)

        self.assertEqual(masked.hologram_exp[2, 2], 0.0)
        self.assertGreater(unmasked.hologram_exp[2, 2], 0.0)

    def test_detected_hologram_can_skip_detector_threshold(self) -> None:
        """Test that detected hologram can skip detector threshold.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        thresholded = self._hologram()
        thresholded.hologram_detector[:] = 100.0
        thresholded.detector_threshold = 10.0
        thresholded.add_noise(
            apply_beamstop_mask=False,
            apply_detector_threshold=True,
        )

        unthresholded = self._hologram()
        unthresholded.hologram_detector[:] = 100.0
        unthresholded.detector_threshold = 10.0
        unthresholded.add_noise(
            apply_beamstop_mask=False,
            apply_detector_threshold=False,
        )

        self.assertEqual(np.max(thresholded.hologram_exp), 10.0)
        self.assertGreater(np.max(unthresholded.hologram_exp), 10.0)

    def test_no_beamstop_variant_reuses_same_noise_realization(self) -> None:
        """Test that no beamstop variant reuses same noise realization.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        holo = self._hologram()
        holo.readout_noise_average = 10.0
        holo.readout_noise_sigma = 2.0
        holo.add_noise(store_no_beamstop=True)

        masked_unblocked = holo.hologram_exp[holo.beamstop.beamstop == 0]
        unmasked_unblocked = holo.hologram_exp_no_beamstop[holo.beamstop.beamstop == 0]

        self.assertTrue(np.array_equal(masked_unblocked, unmasked_unblocked))
        self.assertGreater(holo.hologram_exp[2, 2], 0.0)
        self.assertGreater(holo.hologram_exp_no_beamstop[2, 2], holo.hologram_exp[2, 2])

    def test_hologram_config_stores_no_beamstop_source(self) -> None:
        """Test that hologram config stores no beamstop source.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        config = HologramConfig()
        arr = np.ones((3, 3), dtype=float)
        config.add_holograms({"CR": arr}, source="detected_no_beamstop")

        saved = config.to_dict(
            helicities=["CR"],
            sources=["detected_no_beamstop"],
        )

        self.assertTrue(np.array_equal(saved["CR"]["detected_no_beamstop"], arr[None]))

    def test_gnomonic_projection_preserves_non_negative_intensity(self) -> None:
        """Test that detector projection cannot create negative ideal pixels."""
        layout = detector_layout(
            pixel_size=1.0,
            detector_shape=(33, 33),
            distance_sample_detector=100.0,
            detector_center=(16, 16),
        )
        layout.detqy, layout.detqx = np.meshgrid(
            np.linspace(-0.45, 0.45, 33),
            np.linspace(-0.45, 0.45, 33),
            indexing="ij",
        )
        hologram = np.zeros((33, 33), dtype=float)
        hologram[12:21, 12:21] = 1.0

        projected = detector_hologram(
            detector_layout=layout,
            hologram=hologram,
            beam_parameters=SimpleNamespace(coherence_length=None),
            real_space_pixel_size=33.0,
            beamstop=SimpleNamespace(beamstop=np.zeros((33, 33), dtype=float)),
            measurement_config={"number_frames": 1},
            detector_params={
                "readout_noise_average": 0.0,
                "readout_noise_sigma": 0.0,
            },
        )
        projected.gnomonic_projection()

        self.assertGreaterEqual(np.min(projected.hologram_detector), 0.0)

    def test_gnomonic_projection_can_average_detector_pixel_footprint(self) -> None:
        """Test optional finite-pixel footprint averaging for ideal holograms."""
        layout = detector_layout(
            pixel_size=1.0,
            detector_shape=(17, 17),
            distance_sample_detector=100.0,
            detector_center=(8, 8),
        )
        layout.calc_q_space_coordinates(SimpleNamespace(wavevector=1.0))
        hologram = np.zeros((17, 17), dtype=float)
        hologram[7:10, 7:10] = 1.0

        projected = detector_hologram(
            detector_layout=layout,
            hologram=hologram,
            beam_parameters=SimpleNamespace(wavevector=1.0, coherence_length=None),
            real_space_pixel_size=17.0,
            beamstop=SimpleNamespace(beamstop=np.zeros((17, 17), dtype=float)),
            measurement_config={"number_frames": 1},
            detector_params={
                "readout_noise_average": 0.0,
                "readout_noise_sigma": 0.0,
            },
        )
        result = projected.gnomonic_projection(
            use_pixel_footprint=True,
            pixel_footprint_samples=3,
        )

        self.assertEqual(result.shape, (17, 17))
        self.assertGreaterEqual(np.min(result), 0.0)
        self.assertLessEqual(np.max(result), 1.0)

    def test_gnomonic_projection_applies_flat_detector_solid_angle(self) -> None:
        """Test that off-axis flat detector pixels collect less solid angle."""
        layout = detector_layout(
            pixel_size=1.0,
            detector_shape=(5, 5),
            distance_sample_detector=2.0,
            detector_center=(2, 2),
        )
        layout.detqx = np.zeros((5, 5), dtype=float)
        layout.detqy = np.zeros((5, 5), dtype=float)

        projected = detector_hologram(
            detector_layout=layout,
            hologram=np.ones((5, 5), dtype=float),
            beam_parameters=SimpleNamespace(wavevector=1.0, coherence_length=None),
            real_space_pixel_size=5.0,
            beamstop=SimpleNamespace(beamstop=np.zeros((5, 5), dtype=float)),
            measurement_config={"number_frames": 1},
            detector_params={
                "readout_noise_average": 0.0,
                "readout_noise_sigma": 0.0,
            },
        )
        result = projected.gnomonic_projection()

        expected_corner = (2.0 / np.sqrt(2.0**2 + 2.0**2 + 2.0**2)) ** 3
        self.assertAlmostEqual(result[2, 2], 1.0)
        self.assertAlmostEqual(result[0, 0], expected_corner)
        self.assertLess(result[0, 0], result[2, 2])

    def test_detector_q_coordinates_can_ignore_flat_detector_curvature(self) -> None:
        """Test the optional linear detector-position to q-space mapping."""
        layout = detector_layout(
            pixel_size=1.0,
            detector_shape=(1, 3),
            distance_sample_detector=2.0,
            detector_center=(0, 1),
        )
        beam = SimpleNamespace(wavevector=10.0)

        layout.calc_q_space_coordinates(beam)
        curved = np.asarray(layout.detqx).copy()

        layout.calc_q_space_coordinates(
            beam,
            ignore_flat_detector_curvature=True,
        )
        linear = np.asarray(layout.detqx)

        self.assertAlmostEqual(curved[0, 2], 10.0 * np.sin(np.arctan(0.5)))
        self.assertAlmostEqual(linear[0, 2], 5.0)
        self.assertGreater(linear[0, 2], curved[0, 2])

    def test_linear_projection_broadcasts_sparse_detector_coordinates(self) -> None:
        """Test linear q mapping produces homogeneous map_coordinates input."""
        layout = detector_layout(
            pixel_size=1.0,
            detector_shape=(5, 7),
            distance_sample_detector=10.0,
            detector_center=(2, 3),
        )
        layout.calc_q_space_coordinates(SimpleNamespace(wavevector=1.0))
        projected = detector_hologram(
            detector_layout=layout,
            hologram=np.ones((5, 7), dtype=float),
            beam_parameters=SimpleNamespace(wavevector=1.0, coherence_length=None),
            real_space_pixel_size=10.0,
            beamstop=SimpleNamespace(beamstop=np.zeros((5, 7), dtype=float)),
            measurement_config={"number_frames": 1},
            detector_params={
                "readout_noise_average": 0.0,
                "readout_noise_sigma": 0.0,
            },
        )

        result = projected.gnomonic_projection(
            ignore_flat_detector_curvature=True,
        )

        self.assertEqual(result.shape, (5, 7))
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == "__main__":
    unittest.main()
