"""Tests for illumination wavefield construction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scattering_calculator.experimental_conditions import light_beam


def test_beam_direction_from_alpha_uses_y_x_convention() -> None:
    """Beam tilt helper returns a lab (x, y, z) propagation direction."""
    direction = light_beam.beam_direction_from_alpha((0.1, 0.2))
    expected = np.array([np.tan(0.2), np.tan(0.1), 1.0])
    expected = expected / np.linalg.norm(expected)

    np.testing.assert_allclose(direction, expected)


def test_gauss_beam_zero_tilt_matches_normal_incidence_exactly() -> None:
    """alpha_beam=0 must preserve the historical Gaussian beam field."""
    kwargs = dict(
        sz=(32, 32),
        px_size=4e-9,
        center=np.array([16.0, 16.0]),
        distance=1e-3,
        fwhm=0.5e-6,
        wavelength=1.6e-9,
    )

    normal = light_beam.gauss_beam(**kwargs)
    tilted_zero = light_beam.gauss_beam(**kwargs, alpha_beam=0.0)
    tilted_zero_tuple = light_beam.gauss_beam(**kwargs, alpha_beam=(0.0, 0.0))

    assert np.array_equal(tilted_zero, normal)
    assert np.array_equal(tilted_zero_tuple, normal)


def test_gauss_beam_tilt_changes_phase_and_keeps_shape() -> None:
    """A nonzero beam tilt evaluates the Gaussian on the tilted sample plane."""
    kwargs = dict(
        sz=(32, 32),
        px_size=4e-9,
        center=np.array([16.0, 16.0]),
        distance=1e-3,
        fwhm=0.5e-6,
        wavelength=1.6e-9,
    )

    normal = light_beam.gauss_beam(**kwargs)
    tilted = light_beam.gauss_beam(**kwargs, alpha_beam=(0.0, 0.1))

    assert tilted.shape == normal.shape
    assert np.all(np.isfinite(tilted))
    assert not np.allclose(tilted, normal)


def test_gauss_beam_scalar_tilt_matches_x_direction_tuple() -> None:
    """A scalar alpha_beam remains a backward-compatible x-direction tilt."""
    kwargs = dict(
        sz=(32, 32),
        px_size=4e-9,
        center=np.array([16.0, 16.0]),
        distance=1e-3,
        fwhm=0.5e-6,
        wavelength=1.6e-9,
    )

    scalar = light_beam.gauss_beam(**kwargs, alpha_beam=0.1)
    tuple_x = light_beam.gauss_beam(**kwargs, alpha_beam=(0.0, 0.1))

    assert np.array_equal(scalar, tuple_x)


def test_gauss_beam_tilt_preserves_non_square_shape() -> None:
    """Tilted illumination should follow the requested sample-plane shape."""
    tilted = light_beam.gauss_beam(
        sz=(24, 32),
        px_size=4e-9,
        center=np.array([12.0, 16.0]),
        distance=1e-3,
        fwhm=0.5e-6,
        wavelength=1.6e-9,
        alpha_beam=(0.1, 0.1),
    )

    assert tilted.shape == (24, 32)


def test_gauss_beam_center_is_respected_for_non_square_shape() -> None:
    """Gaussian coordinates should not swap rectangular y/x dimensions."""
    beam = light_beam.gauss_beam(
        sz=(24, 32),
        px_size=4e-9,
        center=np.array([12.0, 16.0]),
        distance=0.0,
        fwhm=0.5e-6,
        wavelength=1.6e-9,
        alpha_beam=(0.0, 0.0),
    )

    assert beam.shape == (24, 32)
    assert np.unravel_index(np.abs(beam).argmax(), beam.shape) == (12, 16)


def test_illumination_gauss_beam_center_uses_x_and_y_shape() -> None:
    """The wrapper should convert centre metres with the correct axis lengths."""
    params = light_beam.beam_parameters(
        photon_energy=778.0,
        pol="CR",
        photon_flux=1.0,
        coherence_length=(1.0, 1.0),
    )
    illumination = light_beam.illumination(
        beam_parameters=params,
        sample_shape=(24, 32),
        real_space_pixel_size=4e-9,
    )

    illumination.gauss_beam(
        center=np.array([0.0, 0.0]),
        distance=0.0,
        fwhm=0.5e-6,
    )
    field = illumination.return_illumination()

    assert field is not None
    assert field.shape == (24, 32)
    assert np.unravel_index(np.abs(field).argmax(), field.shape) == (12, 16)
