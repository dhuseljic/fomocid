"""Streaming variant noise preserves the original event model and edge crops."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal

from scattering_calculator.experimental_conditions.detector import detector_hologram


def reference_splat(counts, bank, classes, rng):
    result = np.zeros(counts.shape, dtype=float)
    for c in range(bank.shape[0]):
        mask = classes == c
        if not np.any(mask):
            continue
        class_counts = np.zeros_like(counts)
        class_counts[mask] = counts[mask]
        if class_counts.sum() == 0:
            continue
        variants = detector_hologram.split_counts_into_variants(None, class_counts, bank.shape[1], rng)
        for v, image in enumerate(variants):
            if image.sum() > 0:
                kernel = bank[c, v] / bank[c, v].sum()
                result += signal.fftconvolve(image, kernel, mode='same')
    return result


@pytest.mark.parametrize('shape,kernel_shape', [((17, 23), (9, 9)), ((16, 20), (4, 6)), ((3, 5), (7, 9))])
@pytest.mark.parametrize('mean', [0., .03, 2., 100.])
@pytest.mark.parametrize('variants', [1, 5])
def test_streaming_splat_matches_seeded_reference_and_rng_state(shape, kernel_shape, mean, variants):
    source = np.random.default_rng(8)
    counts = source.poisson(mean, shape)
    classes = source.integers(0, 3, shape, dtype=np.int32)
    bank = source.uniform(.01, 1, (3, variants, *kernel_shape))
    first, second = np.random.default_rng(7), np.random.default_rng(7)
    expected = reference_splat(counts, bank, classes, first)
    actual = detector_hologram.photon_splat_with_spatial_classes(None, counts, bank, classes, second)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert first.bit_generator.state == second.bit_generator.state


def test_detector_noise_timing_covers_accumulated_and_per_frame():
    from scattering_calculator.experimental_conditions.detector import detector_hologram
    for mode in ['accumulated', 'per_frame']:
        shape = (16, 20)
        model = detector_hologram(SimpleNamespace(detector_shape=shape, detector_center=(8,10)),
            np.ones(shape), SimpleNamespace(energy=778., coherence_length=None), 1.,
            SimpleNamespace(beamstop=np.zeros(shape)), artifacts_config={'sigma_photon':0},
            measurement_config={'number_frames':3,'acquisition_mode':mode},
            detector_params={'noise_seed':7})
        model.hologram_detector = np.ones(shape)
        model.add_noise(store_no_beamstop=True)
        assert model.noise_timings_seconds['total'] > 0
        assert all(value >= 0 for value in model.noise_timings_seconds.values())
