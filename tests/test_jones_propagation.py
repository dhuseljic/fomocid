"""Test Jones propagation padding, absorbers, and ROI propagation."""

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
        """Test that padded propagation preserves original shape.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
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
        """Test that edge absorber is one in the center.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
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
        """Test that absorber profile varies smoothly.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
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
        """Test that compact dielectric stack matches dense stack.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
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

    def test_roi_free_space_propagation_keeps_plane_phase_outside_roi(self) -> None:
        """Test that roi free space propagation keeps plane phase outside roi.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        field = np.ones((16, 18, 2), dtype=complex)
        field[7:9, 8:10, 0] = 5.0
        field[7:9, 8:10, 1] = -2.0j
        region = (slice(6, 11), slice(7, 12))
        roi_padding_px = 2

        wf = object.__new__(wavefronts)
        out = wf.propagate_free_space_jones_roi(
            field,
            wavelength=1e-9,
            dz=4e-9,
            pixel_size=1e-9,
            aperture_support_regions=(region,),
            roi_padding_px=roi_padding_px,
            padding_px=2,
            padding_mode="edge",
        )

        expected_outside = field * np.exp(-1j * 2 * np.pi / 1e-9 * 4e-9)
        outside = np.ones(field.shape[:2], dtype=bool)
        padded_region = (
            slice(region[0].start - roi_padding_px, region[0].stop + roi_padding_px),
            slice(region[1].start - roi_padding_px, region[1].stop + roi_padding_px),
        )
        outside[padded_region] = False

        self.assertEqual(out.shape, field.shape)
        self.assertTrue(np.allclose(out[outside], expected_outside[outside]))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_roi_free_space_propagation_merges_overlapping_roi_crops(self) -> None:
        """Test that overlapping roi crops are propagated as one merged crop.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        field = np.ones((8, 8, 2), dtype=complex)
        wavelength = 1e-9
        dz = 2e-9
        baseline = field * np.exp(-1j * 2 * np.pi / wavelength * dz)
        correction = 2.0 + 0.5j
        calls = []

        def fake_local_propagator(
            E_crop,
            wavelength,
            dz,
            pixel_size,
            padding_px=0,
            padding_mode="edge",
            absorber_width_px=0,
            absorber_strength=0.0,
            absorber_profile="cosine",
        ):
            calls.append(E_crop.shape)
            local_baseline = E_crop * np.exp(-1j * 2 * np.pi / wavelength * dz)
            return local_baseline + correction

        wf = object.__new__(wavefronts)
        wf.propagate_free_space_jones = fake_local_propagator
        regions = (
            (slice(1, 5), slice(1, 5)),
            (slice(3, 7), slice(3, 7)),
        )

        out = wf.propagate_free_space_jones_roi(
            field,
            wavelength=wavelength,
            dz=dz,
            pixel_size=1e-9,
            aperture_support_regions=regions,
        )

        expected = baseline.copy()
        expected[slice(1, 7), slice(1, 7), :] += correction

        self.assertEqual(calls, [(6, 6, 2)])
        self.assertTrue(np.allclose(out, expected))

    def test_roi_free_space_propagation_adds_disjoint_roi_corrections(self) -> None:
        """Test that disjoint roi crops add corrections to the baseline field.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        field = np.ones((8, 8, 2), dtype=complex)
        wavelength = 1e-9
        dz = 2e-9
        baseline = field * np.exp(-1j * 2 * np.pi / wavelength * dz)
        corrections = [2.0 + 0.5j, -0.25 + 1.0j]
        calls = {"count": 0}

        def fake_local_propagator(
            E_crop,
            wavelength,
            dz,
            pixel_size,
            padding_px=0,
            padding_mode="edge",
            absorber_width_px=0,
            absorber_strength=0.0,
            absorber_profile="cosine",
        ):
            correction = corrections[calls["count"]]
            calls["count"] += 1
            local_baseline = E_crop * np.exp(-1j * 2 * np.pi / wavelength * dz)
            return local_baseline + correction

        wf = object.__new__(wavefronts)
        wf.propagate_free_space_jones = fake_local_propagator
        regions = (
            (slice(1, 3), slice(1, 3)),
            (slice(5, 7), slice(5, 7)),
        )

        out = wf.propagate_free_space_jones_roi(
            field,
            wavelength=wavelength,
            dz=dz,
            pixel_size=1e-9,
            aperture_support_regions=regions,
        )

        expected = baseline.copy()
        expected[regions[0][0], regions[0][1], :] += corrections[0]
        expected[regions[1][0], regions[1][1], :] += corrections[1]

        self.assertEqual(calls["count"], 2)
        self.assertTrue(np.allclose(out, expected))

    def test_pad_regions_clips_to_field_shape(self) -> None:
        """Test that pad regions clips to field shape.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        regions = ((slice(1, 4), slice(2, 5)),)
        padded = wavefronts._pad_regions(regions, shape=(6, 7), padding_px=3)

        self.assertEqual(padded, ((slice(0, 6), slice(0, 7)),))

    def test_roi_free_space_propagation_falls_back_without_regions(self) -> None:
        """Test that roi free space propagation falls back without regions.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        field = np.zeros((8, 9, 2), dtype=complex)
        field[4, 4, 0] = 1.0

        wf = object.__new__(wavefronts)
        roi_out = wf.propagate_free_space_jones_roi(
            field,
            wavelength=1e-9,
            dz=3e-9,
            pixel_size=1e-9,
            aperture_support_regions=None,
        )
        full_out = wf.propagate_free_space_jones(
            field,
            wavelength=1e-9,
            dz=3e-9,
            pixel_size=1e-9,
        )

        self.assertTrue(np.allclose(roi_out, full_out))


if __name__ == "__main__":
    unittest.main()
