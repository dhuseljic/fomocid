"""Physical stop-plane propagation preserves complex fields and sampling checks."""
from types import SimpleNamespace

import numpy as np
import pytest

from scattering_calculator.beam_propagator.beamstop_diffraction import (
    propagate_with_stop, stop_plane_mask,
)
from scattering_calculator.beam_propagator.detector_propagation import RayleighSommerfeldPropagator


def test_transparent_stop_matches_direct_propagation_for_complex_channels():
    rng = np.random.default_rng(12)
    field = rng.normal(size=(8, 8, 2)) + 1j*rng.normal(size=(8, 8, 2))
    y, x = (np.indices((8, 8))-4)*.03
    expected = RayleighSommerfeldPropagator((8, 8), .03, .1, 2., x, y).forward(field)
    actual = propagate_with_stop(field, .03, .1, 2., x, y,
                                np.zeros((8, 8)), .03, x, y, 1., 1.)
    np.testing.assert_array_equal(actual, expected)


def test_partial_stop_uses_amplitude_transmission_and_preserves_channels():
    y, x = (np.indices((16, 16))-8)*.02
    field = np.exp(-(x*x+y*y)/.03**2).astype(complex)
    opacity = np.zeros(field.shape)
    opacity[6:10, 6:10] = .75
    direct = RayleighSommerfeldPropagator(field.shape, .02, .1, 2., x, y).forward(field)
    at_stop = RayleighSommerfeldPropagator(field.shape, .02, .1, 1., x, y).forward(field)
    blocked = at_stop * (opacity != 0) * .5  # 1 - sqrt(1-.75)
    expected = direct - RayleighSommerfeldPropagator(field.shape, .02, .1, 1., x, y).forward(blocked)
    channels = np.stack((field, 1j*field), axis=-1)
    actual = propagate_with_stop(channels, .02, .1, 2., x, y, opacity, .02, x, y, 1., 1.)
    np.testing.assert_allclose(actual[..., 0], expected, rtol=1e-13)
    np.testing.assert_allclose(actual[..., 1], 1j*expected, rtol=1e-13)
    assert np.max(np.abs(expected-direct)) > 0


def test_stop_plane_rejects_invalid_distance_and_undersampled_phase():
    cfg = SimpleNamespace(sample_to_detector_distance=2.,
        beamstop_config=SimpleNamespace(bs_detector_distance=2.),
        beamstop_diffraction_shape=(16, 16), beamstop_diffraction_pixel_size=.1,
        detector_layout=SimpleNamespace(detx=np.ones((2, 2)), dety=np.ones((2, 2))))
    with pytest.raises(ValueError, match='beamstop-to-camera'):
        stop_plane_mask(cfg, .01)
    cfg.beamstop_config.bs_detector_distance = 1.
    with pytest.raises(ValueError, match='undersamples phase'):
        stop_plane_mask(cfg, .01)
