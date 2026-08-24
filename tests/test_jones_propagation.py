"""Test Jones propagation padding, absorbers, and ROI propagation."""

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

from scattering_calculator.beam_propagator.Jones_propagator import wavefronts
from scattering_calculator.beam_propagator.free_space_sampling import (
    propagation_sampling_limit,
    required_fixed_grid_substeps,
)
from scattering_calculator.beam_propagator.simple_propagation import (
    calculate_scalar_refractive_index_stack,
    scalar_wavefronts,
)
from scattering_calculator.experimental_conditions import light_beam
from scattering_calculator.sample_generator.structures import CompactDielectricTensorStack


class JonesFreeSpacePropagationTests(unittest.TestCase):
    def test_intermediate_wavefields_are_opt_in(self) -> None:
        field = np.zeros((4, 5, 2), dtype=complex)
        field[..., 0] = 1.0
        eps = np.zeros((3, 4, 5, 2, 2), dtype=complex)
        eps[..., 0, 0] = 1.0
        eps[..., 1, 1] = 1.0
        beam = SimpleNamespace(wavelength=2e-9)

        default = wavefronts(
            beam_parameters=beam, eps_stack=eps,
            layer_thicknesses=[1e-9, 1e-9, 1e-9],
            real_space_pixel_size=1e-9, E_in=field,
            propagate=False, calculate_farfield=False,
        )
        stored = wavefronts(
            beam_parameters=beam, eps_stack=eps,
            layer_thicknesses=[1e-9, 1e-9, 1e-9],
            real_space_pixel_size=1e-9, E_in=field,
            propagate=False, calculate_farfield=False,
            store_intermediate_wavefields=True,
        )

        self.assertIsNone(default.intermediate_wavefields)
        self.assertEqual(stored.intermediate_wavefields.shape, (3, 4, 5, 2))
        np.testing.assert_allclose(stored.intermediate_wavefields[-1], stored.exit_wave)
        self.assertIsNone(stored.detector_wave)
        self.assertIsNone(stored.hologram)

    def test_local_wavevector_directions_follow_phase_gradient(self) -> None:
        """Local k estimation follows the simulator's tilted-phase convention."""
        wavelength = 2.0e-9
        pixel_size = 1.0e-9
        alpha_x = 0.2
        k0 = 2 * np.pi / wavelength
        x = np.arange(9, dtype=float)[None, :] * pixel_size
        scalar = np.exp(-1j * k0 * np.sin(alpha_x) * x)
        field = np.zeros((7, 9, 2), dtype=complex)
        field[..., 0] = scalar

        k_map = wavefronts.local_wavevector_directions(field, wavelength, pixel_size)

        np.testing.assert_allclose(k_map[:, 1:-1, 0], np.sin(alpha_x), atol=2e-2)
        np.testing.assert_allclose(k_map[:, 1:-1, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            k_map[:, 1:-1, 2],
            np.sqrt(1.0 - np.sin(alpha_x) ** 2),
            atol=2e-2,
        )

    def test_linear_polarization_labels_and_legacy_aliases(self) -> None:
        """LH/LV are the preferred linear labels; x/y remain aliases."""
        np.testing.assert_allclose(
            light_beam.polarization_vector("LH"),
            light_beam.polarization_vector("x"),
        )
        np.testing.assert_allclose(
            light_beam.polarization_vector("LV"),
            light_beam.polarization_vector("y"),
        )

    def test_farfield_oversampling_uses_background_extended_field(self) -> None:
        """Test that far-field oversampling FFTs a background-filled field."""
        field = np.zeros((4, 4, 2), dtype=complex)
        field[..., 0] = 2.0
        background = np.ones((8, 8, 2), dtype=complex)
        eps = np.zeros((1, 4, 4, 2, 2), dtype=complex)
        eps[..., 0, 0] = 1.0
        eps[..., 1, 1] = 1.0

        wf = wavefronts(
            beam_parameters=SimpleNamespace(wavelength=1.0),
            eps_stack=eps,
            layer_thicknesses=[0.0],
            real_space_pixel_size=1.0,
            E_in=field,
            farfield_oversampling=2,
            farfield_background_jones=background,
        )

        self.assertEqual(wf.exit_wave.shape, (4, 4, 2))
        self.assertEqual(wf.exit_wave_for_farfield.shape, (8, 8, 2))
        np.testing.assert_allclose(wf.exit_wave_for_farfield[:2, :2, :], 1.0)
        np.testing.assert_allclose(wf.exit_wave_for_farfield[2:6, 2:6, :], wf.exit_wave)
        self.assertEqual(wf.hologram.shape, (8, 8))

    def test_farfield_oversampling_requires_background_field(self) -> None:
        """Test that physical far-field padding refuses implicit zero padding."""
        field = np.zeros((4, 4, 2), dtype=complex)
        eps = np.zeros((1, 4, 4, 2, 2), dtype=complex)
        eps[..., 0, 0] = 1.0
        eps[..., 1, 1] = 1.0

        with self.assertRaisesRegex(ValueError, "farfield_background_jones"):
            wavefronts(
                beam_parameters=SimpleNamespace(wavelength=1.0),
                eps_stack=eps,
                layer_thicknesses=[0.0],
                real_space_pixel_size=1.0,
                E_in=field,
                farfield_oversampling=2,
            )

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

    def test_sampling_limit_uses_transverse_pixel_count(self) -> None:
        limit = propagation_sampling_limit((12, 20), pixel_size=2.0, wavelength=0.5)

        self.assertEqual(limit, 96.0)
        self.assertEqual(
            required_fixed_grid_substeps(
                (12, 20), pixel_size=2.0, wavelength=0.5, dz=192.1
            ),
            3,
        )

    def test_scalar_long_distance_auto_substeps_fixed_grid_propagation(self) -> None:
        rng = np.random.default_rng(4)
        field = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        wavelength = 1.0
        pixel_size = 1.0
        dz = 20.0
        substeps = required_fixed_grid_substeps(
            field.shape, pixel_size, wavelength, dz
        )
        self.assertGreater(substeps, 1)

        wf = scalar_wavefronts.__new__(scalar_wavefronts)
        out = wf.propagate_free_space_scalar(
            field,
            wavelength=wavelength,
            dz=dz,
            pixel_size=pixel_size,
        )

        expected = field
        step_dz = dz / substeps
        for _ in range(substeps):
            H = wf._free_space_kernel(*field.shape, wavelength, step_dz, pixel_size)
            expected = np.fft.ifft2(np.fft.fft2(expected) * H)

        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_single_fft_fresnel_reports_changed_output_sampling(self) -> None:
        field = np.ones((6, 10, 2), dtype=complex)
        wavelength = 2.0
        dz = 30.0
        pixel_size = 0.5

        out, output_pixel_size = wavefronts.propagate_free_space_jones_fresnel_single_fft(
            field,
            wavelength=wavelength,
            dz=dz,
            pixel_size=pixel_size,
        )

        self.assertEqual(out.shape, field.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(
            output_pixel_size,
            (
                wavelength * abs(dz) / (field.shape[0] * pixel_size),
                wavelength * abs(dz) / (field.shape[1] * pixel_size),
            ),
        )

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

    def test_roi_correction_window_tapers_only_artificial_edges(self) -> None:
        """Test that ROI correction windows avoid hard artificial crop edges."""
        window = wavefronts._roi_correction_window(
            (6, 6),
            (10, 10),
            (slice(0, 6), slice(2, 8)),
            taper_px=2,
        )

        self.assertTrue(np.allclose(window[:, 0], 0.0))
        self.assertTrue(np.allclose(window[:, -1], 0.0))
        self.assertAlmostEqual(window[0, 3], 1.0)
        self.assertAlmostEqual(window[3, 3], 1.0)

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

    def test_jones_only_can_apply_zero_order_free_space_phase(self) -> None:
        """Test that Jones-only propagation can keep longitudinal phase."""
        field = np.zeros((5, 6, 2), dtype=complex)
        field[..., 0] = 1.0
        field[..., 1] = 0.5j

        eps = np.zeros((2, 5, 6, 2, 2), dtype=complex)
        eps[..., 0, 0] = 1.0
        eps[..., 1, 1] = 1.0

        wf = object.__new__(wavefronts)
        wf.aperture_support_regions = None
        without_phase = wf.propagate_jones_multislice(
            field,
            eps,
            wavelength=7e-9,
            thicknesses=[3e-9, 5e-9],
            pixel_size=1e-9,
            propagate=False,
            jones_apply_zero_order_phase=False,
        )
        with_phase = wf.propagate_jones_multislice(
            field,
            eps,
            wavelength=7e-9,
            thicknesses=[3e-9, 5e-9],
            pixel_size=1e-9,
            propagate=False,
            jones_apply_zero_order_phase=True,
        )

        expected_phase = np.exp(-1j * 2 * np.pi / 7e-9 * 3e-9)
        np.testing.assert_allclose(with_phase, without_phase * expected_phase)

    def test_scalar_matches_jones_for_diagonal_eigenmode(self) -> None:
        """Scalar propagation matches Jones for a linear diagonal eigenmode."""
        scalar_in = np.ones((5, 6), dtype=complex)
        jones_in = light_beam.scalar_to_jones(scalar_in, "LH")
        eps = np.zeros((2, 5, 6, 2, 2), dtype=complex)
        eps[0, ..., 0, 0] = 4.0
        eps[0, ..., 1, 1] = 5.0
        eps[1, ..., 0, 0] = 6.0
        eps[1, ..., 1, 1] = 7.0
        beam = SimpleNamespace(wavelength=3e-9, pol="LH")

        jones = wavefronts(
            beam_parameters=beam,
            eps_stack=eps,
            layer_thicknesses=[1e-9, 2e-9],
            real_space_pixel_size=1e-9,
            E_in=jones_in,
            propagate=False,
            jones_apply_zero_order_phase=False,
        )
        scalar = scalar_wavefronts(
            beam_parameters=beam,
            eps_stack=eps,
            layer_thicknesses=[1e-9, 2e-9],
            real_space_pixel_size=1e-9,
            E_in=jones_in,
            propagate=False,
            scalar_apply_zero_order_phase=False,
        )

        np.testing.assert_allclose(jones.exit_wave[..., 0], scalar.exit_wave)
        np.testing.assert_allclose(jones.exit_wave[..., 1], 0.0)
        np.testing.assert_allclose(jones.hologram, scalar.hologram)

    def test_scalar_index_stack_uses_refractive_indices_and_mask(self) -> None:
        """Scalar propagation stack is built without dielectric tensors."""
        sample = SimpleNamespace()
        sample.mask = np.ones((1, 3, 4), dtype=float)
        sample.magnetization = np.zeros((1, 3, 4, 3), dtype=float)
        sample.magnetization[0, :, :, 2] = 0.5
        sample.layer_refractive_indices = np.array(
            [[2.0 + 0.1j, 0.2 + 0.03j, 0.0j]],
            dtype=complex,
        )

        stack = calculate_scalar_refractive_index_stack(
            sample,
            "CR",
            use_aperture_roi=False,
            lazy=False,
        )
        dense = stack.materialize()

        expected_material = (
            sample.layer_refractive_indices[0, 0]
            + 0.5 * sample.layer_refractive_indices[0, 1]
        )
        np.testing.assert_allclose(dense[0, 0, 0], expected_material)
        self.assertFalse(hasattr(sample, "final_dielectric_tensor"))

        sample.mask[0, 1, 2] = 0.0
        stack = calculate_scalar_refractive_index_stack(
            sample,
            "CR",
            use_aperture_roi=False,
            lazy=False,
        )
        np.testing.assert_allclose(stack.materialize()[0, 1, 2], 1.0)

    def test_lazy_scalar_index_stack_matches_precomputed_stack(self) -> None:
        """Lazy scalar patches match the precomputed compact representation."""
        sample = SimpleNamespace()
        sample.mask = np.ones((2, 4, 5), dtype=float)
        sample.mask[:, 1:3, 2:4] = 0.4
        sample.magnetization = np.zeros((2, 4, 5, 3), dtype=float)
        sample.magnetization[:, 1:3, 2:4, 2] = 0.7
        sample.layer_refractive_indices = np.array(
            [[2.0 + 0.1j, 0.03j, 0.0], [1.5 + 0.05j, -0.02j, 0.0]],
            dtype=complex,
        )

        lazy_stack = calculate_scalar_refractive_index_stack(
            sample,
            "CR",
            use_aperture_roi=True,
            lazy=True,
        )
        precomputed_stack = calculate_scalar_refractive_index_stack(
            sample,
            "CR",
            use_aperture_roi=True,
            lazy=False,
        )

        self.assertTrue(hasattr(lazy_stack, "layer_patches"))
        np.testing.assert_allclose(
            lazy_stack.materialize(),
            precomputed_stack.materialize(),
        )

    def test_structure_can_store_final_scalar_refractive_index_stack(self) -> None:
        """Structure exposes a scalar analogue to final_dielectric_tensor."""
        sample = SimpleNamespace()
        sample.mask = np.ones((1, 2, 2), dtype=float)
        sample.magnetization = np.zeros((1, 2, 2, 3), dtype=float)
        sample.layer_refractive_indices = np.array(
            [[1.5 + 0.01j, 0.0j, 0.0j]],
            dtype=complex,
        )
        from scattering_calculator.sample_generator.structures import Structure

        Structure.calculate_final_scalar_refractive_index(
            sample,
            "LH",
            use_aperture_roi=False,
            compact=True,
        )

        self.assertTrue(hasattr(sample, "final_scalar_refractive_index"))
        self.assertEqual(sample.final_scalar_refractive_index.shape, (1, 2, 2))
        np.testing.assert_allclose(
            sample.final_scalar_refractive_index.materialize(),
            np.full((1, 2, 2), 1.5 + 0.01j),
        )

    def test_scalar_matches_jones_for_circular_eigenmode(self) -> None:
        """Scalar propagation matches Jones for a circular eigenpolarization."""
        scalar_in = np.ones((4, 5), dtype=complex)
        jones_in = light_beam.scalar_to_jones(scalar_in, "CR")
        eps = np.zeros((1, 4, 5, 2, 2), dtype=complex)
        eps[..., 0, 0] = 4.0
        eps[..., 1, 1] = 4.0
        eps[..., 0, 1] = 0.2j
        eps[..., 1, 0] = -0.2j
        beam = SimpleNamespace(wavelength=2e-9, pol="CR")

        jones = wavefronts(
            beam_parameters=beam,
            eps_stack=eps,
            layer_thicknesses=[1.5e-9],
            real_space_pixel_size=1e-9,
            E_in=jones_in,
            propagate=False,
        )
        scalar = scalar_wavefronts(
            beam_parameters=beam,
            eps_stack=eps,
            layer_thicknesses=[1.5e-9],
            real_space_pixel_size=1e-9,
            E_in=jones_in,
            propagate=False,
        )

        expected_jones = light_beam.scalar_to_jones(scalar.exit_wave, "CR")
        np.testing.assert_allclose(jones.exit_wave, expected_jones)
        np.testing.assert_allclose(jones.hologram, scalar.hologram, atol=1e-10)

    def test_scalar_roi_free_space_matches_jones_for_diagonal_eigenmode(self) -> None:
        """Scalar ROI free-space propagation follows Jones for a linear eigenmode."""
        scalar_in = np.ones((16, 18), dtype=complex)
        scalar_in[7:9, 8:10] = 4.0
        jones_in = light_beam.scalar_to_jones(scalar_in, "LH")
        eps = np.zeros((2, 16, 18, 2, 2), dtype=complex)
        eps[..., 0, 0] = 1.0
        eps[..., 1, 1] = 1.0
        region = (slice(6, 11), slice(7, 12))
        beam = SimpleNamespace(wavelength=1e-9, pol="LH")

        jones = wavefronts(
            beam_parameters=beam,
            eps_stack=eps,
            layer_thicknesses=[0.0, 3e-9],
            real_space_pixel_size=1e-9,
            E_in=jones_in,
            aperture_support_regions=(region,),
            propagate=True,
            propagation_padding_px=2,
            multislice_propagation_roi=True,
            multislice_propagation_roi_padding_px=2,
        )
        scalar = scalar_wavefronts(
            beam_parameters=beam,
            eps_stack=eps,
            layer_thicknesses=[0.0, 3e-9],
            real_space_pixel_size=1e-9,
            E_in=jones_in,
            aperture_support_regions=(region,),
            propagate=True,
            propagation_padding_px=2,
            multislice_propagation_roi=True,
            multislice_propagation_roi_padding_px=2,
        )

        np.testing.assert_allclose(jones.exit_wave[..., 0], scalar.exit_wave)
        np.testing.assert_allclose(jones.exit_wave[..., 1], 0.0)

    def test_scalar_full_free_space_matches_jones_for_linear_eigenmode(self) -> None:
        """Scalar and Jones use the same free-space propagation convention."""
        rng = np.random.default_rng(3)
        scalar_in = rng.normal(size=(12, 14)) + 1j * rng.normal(size=(12, 14))
        jones_in = light_beam.scalar_to_jones(scalar_in, "LH")
        eps = np.zeros((2, 12, 14, 2, 2), dtype=complex)
        eps[..., 0, 0] = 1.0
        eps[..., 1, 1] = 1.0
        beam = SimpleNamespace(wavelength=2e-9, pol="LH")

        jones = wavefronts(
            beam_parameters=beam,
            eps_stack=eps,
            layer_thicknesses=[0.0, 4e-9],
            real_space_pixel_size=1e-9,
            E_in=jones_in,
            propagate=True,
            propagation_padding_px=2,
        )
        scalar = scalar_wavefronts(
            beam_parameters=beam,
            eps_stack=eps,
            layer_thicknesses=[0.0, 4e-9],
            real_space_pixel_size=1e-9,
            E_in=jones_in,
            propagate=True,
            propagation_padding_px=2,
        )

        np.testing.assert_allclose(jones.exit_wave[..., 0], scalar.exit_wave)
        np.testing.assert_allclose(jones.exit_wave[..., 1], 0.0)

    def test_scalar_roi_free_space_keeps_uniform_phase_continuous(self) -> None:
        """Scalar ROI propagation does not introduce a plane-wave phase jump."""
        field = np.ones((14, 16), dtype=complex)
        region = (slice(4, 9), slice(5, 11))
        wf = scalar_wavefronts.__new__(scalar_wavefronts)

        out = wf.propagate_free_space_scalar_roi(
            field,
            wavelength=2e-9,
            dz=5e-9,
            pixel_size=1e-9,
            aperture_support_regions=(region,),
            roi_padding_px=2,
            padding_px=2,
        )

        expected = np.exp(-1j * 2 * np.pi / 2e-9 * 5e-9)
        np.testing.assert_allclose(out, expected)

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

    def test_roi_free_space_propagation_merge_true_uses_common_crop(self) -> None:
        """Test that merge-overlaps propagates all ROI boxes as one crop.

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
            (slice(1, 3), slice(1, 3)),
            (slice(5, 7), slice(5, 7)),
        )

        out = wf.propagate_free_space_jones_roi(
            field,
            wavelength=wavelength,
            dz=dz,
            pixel_size=1e-9,
            aperture_support_regions=regions,
            merge_overlaps=True,
        )

        expected = baseline.copy()
        expected[slice(1, 7), slice(1, 7), :] += correction

        self.assertEqual(calls, [(6, 6, 2)])
        self.assertTrue(np.allclose(out, expected))

    def test_roi_free_space_propagation_merges_overlapping_roi_crops(self) -> None:
        """Test that overlapping roi crops are propagated as one merged crop."""
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
            merge_overlaps=False,
        )

        expected = baseline.copy()
        expected[slice(1, 7), slice(1, 7), :] += correction

        self.assertEqual(calls, [(6, 6, 2)])
        self.assertTrue(np.allclose(out, expected))

    def test_roi_free_space_propagation_merges_physical_overlaps_even_when_disabled(
        self,
    ) -> None:
        """Test that overlapping aperture supports are always merged.

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
            merge_overlaps=False,
        )

        expected = baseline.copy()
        expected[slice(1, 7), slice(1, 7), :] += correction

        self.assertEqual(calls, [(6, 6, 2)])
        self.assertTrue(np.allclose(out, expected))

    def test_roi_free_space_propagation_merges_overlapping_padded_crops(self) -> None:
        """Test that overlapping padded crops merge even when opt-out is set."""
        field = np.ones((10, 10, 2), dtype=complex)
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
            (slice(1, 3), slice(1, 3)),
            (slice(5, 7), slice(5, 7)),
        )

        out = wf.propagate_free_space_jones_roi(
            field,
            wavelength=wavelength,
            dz=dz,
            pixel_size=1e-9,
            aperture_support_regions=regions,
            roi_padding_px=2,
            merge_overlaps=False,
        )

        expected = baseline.copy()
        merged_region = (slice(0, 9), slice(0, 9))
        window = wavefronts._roi_correction_window(
            (9, 9),
            field.shape[:2],
            merged_region,
            taper_px=2,
        )
        expected[merged_region[0], merged_region[1], :] += correction * window[..., None]

        self.assertEqual(calls, [(9, 9, 2)])
        self.assertTrue(np.allclose(out, expected))

    def test_roi_free_space_propagation_keeps_disjoint_padded_crops_separate(self) -> None:
        """Test that non-overlapping padded crops can stay separate."""
        field = np.ones((12, 12, 2), dtype=complex)
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
            (slice(8, 10), slice(8, 10)),
        )

        out = wf.propagate_free_space_jones_roi(
            field,
            wavelength=wavelength,
            dz=dz,
            pixel_size=1e-9,
            aperture_support_regions=regions,
            roi_padding_px=1,
            merge_overlaps=False,
        )

        expected = baseline.copy()
        padded_regions = (
            (slice(0, 4), slice(0, 4)),
            (slice(7, 11), slice(7, 11)),
        )
        first_window = wavefronts._roi_correction_window(
            (4, 4),
            field.shape[:2],
            padded_regions[0],
            taper_px=1,
        )
        second_window = wavefronts._roi_correction_window(
            (4, 4),
            field.shape[:2],
            padded_regions[1],
            taper_px=1,
        )
        expected[padded_regions[0][0], padded_regions[0][1], :] += (
            corrections[0] * first_window[..., None]
        )
        expected[padded_regions[1][0], padded_regions[1][1], :] += (
            corrections[1] * second_window[..., None]
        )

        self.assertEqual(calls["count"], 2)
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
            merge_overlaps=False,
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
