"""Tests for the Mueller--Stokes propagation option and conventions."""

import numpy as np
import pytest

from scattering_calculator.beam_propagator.Stokes_propagator import (
    apply_mueller,
    coherency_to_stokes,
    jones_to_mueller,
    jones_to_stokes,
    stokes_to_coherency,
    stokes_wavefronts,
    validate_stokes,
)


def test_project_polarization_convention():
    horizontal = np.array([1.0, 0.0], dtype=complex)
    right_circular = np.array([1.0, -1j], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(jones_to_stokes(horizontal), [1, 1, 0, 0])
    np.testing.assert_allclose(jones_to_stokes(right_circular), [1, 0, 0, 1])


def test_coherency_round_trip_supports_partially_polarized_light():
    stokes = np.array([2.0, 0.3, -0.4, 0.5])
    np.testing.assert_allclose(coherency_to_stokes(stokes_to_coherency(stokes)), stokes)


def test_jones_and_mueller_transformations_agree():
    field = np.array([0.7 + 0.2j, -0.1 + 0.8j])
    jones = np.array([[0.8, 0.2j], [-0.1j, 0.6]], dtype=complex)
    expected = jones_to_stokes(jones @ field)
    actual = apply_mueller(jones_to_stokes(field), jones_to_mueller(jones))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_depolarizing_mueller_matrix_is_supported():
    # An ideal depolarizer keeps total intensity and erases Q, U, and V.  It
    # cannot be represented by a single Jones matrix, which is exactly why the
    # standalone Mueller operation is useful.
    depolarizer = np.diag([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        apply_mueller(np.array([1.0, 0.2, 0.3, 0.4]), depolarizer),
        [1.0, 0.0, 0.0, 0.0],
    )


def test_stokes_validation_rejects_degree_of_polarization_above_one():
    validate_stokes(np.array([1.0, 0.2, 0.3, 0.4]))
    with pytest.raises(ValueError, match="Unphysical Stokes"):
        validate_stokes(np.array([1.0, 1.1, 0.0, 0.0]))


def test_stokes_wavefront_matches_jones_intensity_for_identity_slice():
    class Beam:
        wavelength = 1e-9

    incoming = np.ones((4, 5, 2), dtype=complex) / np.sqrt(2)
    eps = np.zeros((1, 4, 5, 2, 2), dtype=complex)
    eps[..., 0, 0] = 1.0
    eps[..., 1, 1] = 1.0
    wavefront = stokes_wavefronts(
        beam_parameters=Beam(),
        eps_stack=eps,
        layer_thicknesses=[0.0],
        real_space_pixel_size=1e-7,
        E_in=incoming,
    )
    assert wavefront.exit_wave.shape == (4, 5, 4)
    np.testing.assert_allclose(wavefront.exit_stokes[..., 0], 1.0)
    np.testing.assert_allclose(
        wavefront.hologram,
        np.sum(np.abs(wavefront.detector_jones) ** 2, axis=-1),
    )


def test_stokes_wavefront_accepts_partially_polarized_input():
    class Beam:
        wavelength = 1e-9

    # The template supplies the real-space illumination envelope only.  The
    # explicit input Stokes vector below replaces its polarization by a 60 %
    # right-circular component plus a 40 % unpolarized component.
    incoming_template = np.ones((4, 5, 2), dtype=complex) / np.sqrt(2)
    eps = np.zeros((1, 4, 5, 2, 2), dtype=complex)
    eps[..., 0, 0] = 1.0
    eps[..., 1, 1] = 1.0

    wavefront = stokes_wavefronts(
        beam_parameters=Beam(),
        eps_stack=eps,
        layer_thicknesses=[0.0],
        real_space_pixel_size=1e-7,
        E_in=incoming_template,
        input_stokes=np.array([1.0, 0.0, 0.0, 0.6]),
    )

    assert len(wavefront._coherent_modes) == 2
    np.testing.assert_allclose(wavefront.input_stokes[..., 0], 1.0)
    np.testing.assert_allclose(wavefront.input_stokes[..., 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(wavefront.input_stokes[..., 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(wavefront.input_stokes[..., 3], 0.6)
    np.testing.assert_allclose(wavefront.hologram, wavefront.detector_stokes[..., 0])
