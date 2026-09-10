"""Physical and integration checks for finite-distance detector propagation."""

from types import SimpleNamespace

import numpy as np
import pytest

from scattering_calculator.beam_propagator.detector_propagation import RayleighSommerfeldPropagator
from scattering_calculator.beam_propagator.Jones_propagator import wavefronts
from scattering_calculator.simulation_pipelines.simulation_configuration import DetectorConfig


def test_adjoint_on_unequal_grids_with_channels():
    rng = np.random.default_rng(83)
    x, y = np.meshgrid(np.linspace(-3, 2, 7), np.linspace(-1, 4, 5))
    op = RayleighSommerfeldPropagator((8, 9), .13, .7, 1.8, x, y,
                                    max_kernel_elements=25)
    source = rng.normal(size=(8, 9, 2)) + 1j * rng.normal(size=(8, 9, 2))
    target = rng.normal(size=(5, 7, 2)) + 1j * rng.normal(size=(5, 7, 2))
    np.testing.assert_allclose(np.vdot(op.forward(source), target),
                               np.vdot(source, op.adjoint(target)), rtol=1e-13)


def test_sparse_source_matches_full_matrix_and_zero_field():
    x, y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-.5, .5, 3))
    op = RayleighSommerfeldPropagator((6, 8), .1, .6, 1.2, x, y,
                                    max_kernel_elements=16)
    source = np.zeros((6, 8, 2), complex)
    source[2, 3, 0] = 1 + 2j
    source[4, 5, 1] = 1e-20  # Small nonzero amplitudes must not be discarded.
    expected = np.zeros((x.size, 2), complex)
    for ds, ss, kernel in op._blocks():
        expected[ds] += kernel @ source.reshape(-1, 2)[ss]
    np.testing.assert_allclose(op.forward(source).reshape(-1, 2), expected, rtol=1e-13, atol=0)
    np.testing.assert_array_equal(op.forward(np.zeros_like(source)), 0)


def test_circular_aperture_against_analytic_on_axis_integral():
    n, pitch, radius, z, wavelength = 200, .01, .7, 1.1, .8
    y, x = (np.indices((n, n)) - n / 2) * pitch
    aperture = (x*x + y*y <= radius*radius).astype(complex)
    op = RayleighSommerfeldPropagator(aperture.shape, pitch, wavelength, z,
                                    np.zeros((1, 1)), np.zeros((1, 1)))
    edge = np.hypot(z, radius)
    k = 2 * np.pi / wavelength
    expected = np.exp(-1j*k*z) - z/edge * np.exp(-1j*k*edge)
    np.testing.assert_allclose(op.forward(aperture)[0, 0], expected, atol=.004)


def test_matches_nonparaxial_angular_spectrum_for_localized_wave():
    n, pitch, wavelength, z = 96, .12, 1., 1.3
    y, x = (np.indices((n, n)) - n / 2) * pitch
    source = np.exp(-(x*x+y*y)/.7**2) * np.exp(1j*3*x)
    padded = np.pad(source, n * 2)
    kernel = wavefronts._free_space_kernel(*padded.shape, wavelength, z, pitch)
    expected = np.fft.ifft2(np.fft.fft2(padded) * kernel)[2*n:3*n, 2*n:3*n]
    sl = np.s_[44:53:2, 44:53:2]
    op = RayleighSommerfeldPropagator(source.shape, pitch, wavelength, z, x[sl], y[sl])
    # The FFT reference has periodic replicas; the RS source has zero exterior.
    np.testing.assert_allclose(op.forward(source), expected[sl], atol=.003)


def test_farfield_limit_has_documented_inverse_fourier_phase():
    n, pitch, wavelength, z = 8, .1, .5, 1e5
    rng = np.random.default_rng(3)
    source = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
    y, x = (np.indices((n, n)) - n/2)*pitch
    X, Y = np.meshgrid(np.array([-.001, 0, .001])*z, np.array([0.])*z)
    op = RayleighSommerfeldPropagator(source.shape, pitch, wavelength, z, X, Y)
    k = 2*np.pi/wavelength
    R = np.sqrt(X*X+Y*Y+z*z)
    expected = np.empty(X.shape, complex)
    for index in np.ndindex(X.shape):
        spectrum = np.sum(source*np.exp(1j*k*(X[index]*x+Y[index]*y)/R[index]))
        expected[index] = 1j/wavelength*z/R[index]**2*np.exp(-1j*k*R[index])*pitch**2*spectrum
    np.testing.assert_allclose(op.forward(source), expected, rtol=5e-5)


def make_detector(field, footprint=False):
    cfg = DetectorConfig(shape=(3, 4), pixel_size=.2, detector_center=(1, 2),
                         sample_to_detector_distance=2.,
                         detector_propagation_method="rayleigh_sommerfeld",
                         use_detector_pixel_footprint=footprint)
    cfg.setup()
    cfg.beamstop = None
    beam = SimpleNamespace(wavelength=.6, coherence_length=None)
    wave = SimpleNamespace(exit_wave=field, hologram=np.zeros(field.shape[:2]))
    prop = SimpleNamespace(return_wavefront=lambda: wave, propagator_method="Scalar",
                           IlluminationConfig=SimpleNamespace(beam_params=beam),
                           SampleConfig=SimpleNamespace(real_space_pixel_size=.1))
    cfg.assign_propagated_wavefront(prop)
    return cfg


@pytest.mark.parametrize("footprint", [False, True])
def test_detector_uses_physical_area_and_preserves_absolute_scale(footprint):
    field = np.ones((4, 6), complex)
    cfg = make_detector(field, footprint)
    cfg.detect_hologram()
    n = 3 if footprint else 1
    expected = np.zeros(cfg.shape)
    for oy in ((np.arange(n)+.5)/n-.5)*.2:
        for ox in ((np.arange(n)+.5)/n-.5)*.2:
            op = RayleighSommerfeldPropagator(field.shape, .1, .6, 2.,
                                            cfg.detector_layout.detx+ox,
                                            cfg.detector_layout.dety+oy)
            expected += abs(op.forward(field))**2 * 4/n**2
    np.testing.assert_allclose(cfg.return_ideal_hologram(), expected)
    brighter = make_detector(2*field, footprint)
    brighter.detect_hologram()
    np.testing.assert_allclose(brighter.return_ideal_hologram(), 4*expected)


def test_configuration_rejects_incompatible_model_and_stokes():
    with pytest.raises(ValueError, match="Unknown detector"):
        DetectorConfig(detector_propagation_method="typo")
    with pytest.raises(ValueError, match="physical detector"):
        DetectorConfig(detector_propagation_method="rayleigh_sommerfeld",
                       ignore_flat_detector_curvature=True)
    cfg = make_detector(np.ones((2, 2), complex))
    cfg.propagator.propagator_method = "Stokes"
    with pytest.raises(ValueError, match="coherent"):
        cfg.assign_propagated_wavefront(cfg.propagator)


def test_stokes_carrier_intensities_add_without_cross_terms():
    field = np.ones((4, 6), complex)
    scalar = make_detector(field)
    scalar.detect_hologram()
    cfg = make_detector(field)
    cfg.propagator.propagator_method = "Stokes"
    # Opposite carrier phases must not cancel: these modes are incoherent.
    cfg.wavefront._coherent_modes = [SimpleNamespace(exit_wave=field),
                                     SimpleNamespace(exit_wave=-field)]
    cfg.assign_propagated_wavefront(cfg.propagator)
    cfg.detect_hologram()
    np.testing.assert_allclose(cfg.return_ideal_hologram(), 2*scalar.return_ideal_hologram())


def test_pipeline_selection_and_metadata(tmp_path):
    from scattering_calculator.simulation_pipelines.pipelines.hologram_pipeline import (
        HologramPipeline, HologramPipelineConfig, HologramPipelineRanges,
    )
    pipeline = HologramPipeline(
        output_path=tmp_path / "wide.h5", n_samples=1,
        config=HologramPipelineConfig(recipe="SiN(80)", detector_propagation_method="rayleigh_sommerfeld"),
        ranges=HologramPipelineRanges(),
    )
    assert pipeline._sample_params()["detector_propagation_method"] == "rayleigh_sommerfeld"
    cfg = make_detector(np.ones((2, 2)))
    assert pipeline._detector_metadata(cfg)["detector/detector_propagation_method"] == "rayleigh_sommerfeld"


def test_jones_channels_sum_incoherently_at_detector():
    field = np.ones((4, 6), complex)
    scalar = make_detector(field)
    scalar.detect_hologram()
    jones = make_detector(np.stack([field, 2j*field], axis=-1))
    jones.propagator.propagator_method = "Jones"
    jones.detect_hologram()
    np.testing.assert_allclose(jones.return_ideal_hologram(), 5*scalar.return_ideal_hologram())


@pytest.mark.parametrize("wavelength", [.4, .8, 1.2])
def test_wide_angle_single_source_matches_green_function_derivative(wavelength):
    # A single occupied quadrature cell at the origin, observed at 60 degrees.
    source = np.zeros((2, 2), complex)
    source[1, 1] = 1
    z, pitch = 2., .01
    X, Y = np.array([[0., z*np.sqrt(3)]]), np.zeros((1, 2))
    op = RayleighSommerfeldPropagator(source.shape, pitch, wavelength, z, X, Y)
    # Independent central difference of -d/dz [exp(-ikr)/(2pi r)].
    k, dz = 2*np.pi/wavelength, 1e-6
    def green(height):
        r = np.sqrt(X**2 + height**2)
        return np.exp(-1j*k*r)/(2*np.pi*r)
    expected = -(green(z+dz)-green(z-dz))/(2*dz)*pitch**2
    np.testing.assert_allclose(op.forward(source), expected, rtol=1e-8)


@pytest.mark.parametrize("dz", [0., .7, -.7])
@pytest.mark.parametrize("frequency", [.95, 1., 1.05])
def test_angular_spectrum_cutoff_and_evanescent_reversal(frequency, dz):
    """Check Paganin's dispersion cutoff and our damped reversal convention."""
    from scattering_calculator.beam_propagator.simple_propagation import scalar_wavefronts

    # The fundamental FFT bin has exactly the requested frequency; lambda=1.
    n = 16
    pitch = 1 / (n * frequency)
    if frequency <= 1:
        expected = np.exp(-2j * np.pi * dz * np.sqrt(1 - frequency**2))
    else:
        expected = np.exp(-2 * np.pi * abs(dz) * np.sqrt(frequency**2 - 1))
    for model in (scalar_wavefronts, wavefronts):
        kernel = model._free_space_kernel(n, n, 1., dz, pitch)
        np.testing.assert_allclose(kernel[0, 1], expected, atol=1e-12)
