"""Photon-unit statistics, energy-dependent gain and per-frame nonlinear response."""
from types import SimpleNamespace
import json

import h5py
import numpy as np
import pytest

from scattering_calculator.experimental_conditions.detector import detector_hologram
from scattering_calculator import simulation_pipelines as sim
from scattering_calculator.simulation_pipelines.examples import magnon_experiment


def energy_gain(energy_eV):
    """Supply an illustrative calibrated energy-dependent gain.

    Parameters
    ----------
    energy_eV : float
        Physical photon energy.

    Returns
    -------
    float
        ADU per detected photon.
    """
    return energy_eV / 100.0


def compress_signal(linear_adu):
    """Supply an illustrative pixelwise charge-compression curve.

    Parameters
    ----------
    linear_adu : ndarray
        Accumulated per-frame linear signal.

    Returns
    -------
    ndarray
        Nonlinear ADU before readout noise and ADC clipping.
    """
    return linear_adu / (1.0 + linear_adu / 1000.0)


def detector_model(
    gain=1.0, response=None, frames=1, rate=100.0, energy=500.0, shape=(12, 14)
):
    """Construct one isolated detector without optical-propagation ambiguity.

    Parameters
    ----------
    gain : float or callable
        ADU per photon.
    response : callable or None
        Optional conversion of total per-frame linear signal.
    frames : int
        Number of independently integrated frames.
    rate : float
        Incident photons per pixel per second.
    energy : float or None
        Physical photon energy in eV.
    shape : tuple
        Detector image shape.

    Returns
    -------
    detector_hologram
        Detector with noise-free electronics and seeded photon statistics.
    """
    image = np.full(shape, rate, dtype=float)
    model = detector_hologram(
        SimpleNamespace(
            detector_shape=shape, detector_center=(shape[0] // 2, shape[1] // 2)
        ),
        image,
        SimpleNamespace(energy=energy, coherence_length=None),
        1.0,
        SimpleNamespace(beamstop=np.zeros(shape)),
        artifacts_config={"sigma_photon": 0.0},
        measurement_config={
            "exposure_time": 1.0,
            "number_frames": frames,
            "max_counts_per_image": None,
        },
        detector_params={
            "counts_per_photon": gain,
            "nonlinear_response": response,
            "quantum_efficiency": 1.0,
            "noise_seed": 123,
            "readout_noise_average": 0.0,
            "readout_noise_sigma": 0.0,
            "detector_threshold": 1e12,
        },
    )
    model.hologram_detector = image.copy()
    return model


def test_gain_scales_adu_and_noise_without_changing_photon_statistics():
    """Verify gain does not divide the physical photon mean before Poisson sampling.

    Parameters
    ----------
    None
        Uses the same photon seed at two gains.

    Returns
    -------
    None
        Checks exact signal scaling and the photon noise mean/variance.
    """
    unit = detector_model(gain=1.0, shape=(200, 200))
    double = detector_model(gain=2.0, shape=(200, 200))
    unit.add_noise()
    double.add_noise()
    np.testing.assert_array_equal(double.hologram_exp, 2 * unit.hologram_exp)
    assert abs(unit.hologram_exp.mean() - 100.0) < 0.3
    assert abs(unit.hologram_exp.var() - 100.0) < 3.0


def test_constant_and_callable_gain_agree_at_one_energy():
    """Keep the scalar ADU/photon interface equivalent to a calibrated function.

    Parameters
    ----------
    None
        Uses 500 eV photons and five ADU per photon.

    Returns
    -------
    None
        Checks identical seeded frames.
    """
    constant = detector_model(5.0)
    function = detector_model(energy_gain)
    constant.add_noise()
    function.add_noise()
    np.testing.assert_array_equal(constant.hologram_exp, function.hologram_exp)


def test_mixed_energies_are_sampled_before_gain_and_one_shared_nonlinearity():
    """Check independent color statistics and combined charge conversion.

    Parameters
    ----------
    None
        Uses two physical photon energies on the same pixels.

    Returns
    -------
    None
        Compares with independently computed Poisson samples and response.
    """
    model = detector_model(energy_gain, compress_signal)
    model.photon_rate_components = (
        (500.0, np.full((12, 14), 100.0)),
        (1000.0, np.full((12, 14), 50.0)),
    )
    rng = np.random.default_rng(123)
    low = rng.poisson(np.full((12, 14), 100.0))
    high = rng.poisson(np.full((12, 14), 50.0))
    expected = np.round(compress_signal(5 * low + 10 * high))
    separate = np.round(compress_signal(5 * low) + compress_signal(10 * high))
    model.add_noise()
    np.testing.assert_array_equal(model.hologram_exp, expected)
    assert not np.array_equal(expected, separate)


def test_nonlinearity_and_saturation_are_applied_before_frame_averaging():
    """Check the per-frame response rather than a curve of accumulated frames.

    Parameters
    ----------
    None
        Uses a quadratic synthetic response and a low ADC threshold.

    Returns
    -------
    None
        Checks per-frame Poisson sampling, conversion, clipping and averaging.
    """
    model = detector_model(2.0, lambda counts: counts**2, frames=3, rate=4.0)
    model.acquisition_mode = "per_frame"
    model.detector_threshold = 70.0
    rng = np.random.default_rng(123)
    expected = []
    for _ in range(3):
        seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        photons = np.random.default_rng(seed).poisson(np.full((12, 14), 4.0))
        expected.append(np.minimum((2 * photons) ** 2, 70.0))
    model.add_noise(store_no_beamstop=True)
    np.testing.assert_array_equal(model.hologram_exp, np.mean(expected, axis=0))
    assert np.all(model.hologram_exp_no_beamstop >= model.hologram_exp)


def test_display_budget_is_defined_before_nonlinear_conversion():
    """Preserve an explicit synthetic ADU budget while sampling real photon numbers.

    Parameters
    ----------
    None
        Uses a normalized display acquisition instead of physical exposure scaling.

    Returns
    -------
    None
        Checks expected photon means and linear ADU budget.
    """
    model = detector_model(5.0)
    model.max_counts_per_image = 200.0
    gain, expected = model._expected_photon_components()[0]
    np.testing.assert_array_equal(gain * expected, np.full((12, 14), 200.0))


@pytest.mark.parametrize(
    "gain", [0.0, -1.0, np.nan, lambda e: np.inf, lambda e: np.array([1.0, 2.0])]
)
def test_invalid_gain_is_rejected(gain):
    """Reject invalid ADU/photon calibration outputs.

    Parameters
    ----------
    gain : object
        Invalid scalar or callback.

    Returns
    -------
    None
        Checks explicit errors before acquisition is used.
    """
    with pytest.raises(ValueError, match="counts_per_photon"):
        detector_model(gain).add_noise()


@pytest.mark.parametrize(
    "response", [lambda s: -s, lambda s: s * np.nan, lambda s: np.zeros((2, 3))]
)
def test_invalid_nonlinear_response_is_rejected(response):
    """Reject response curves with invalid values or incompatible pixel shapes.

    Parameters
    ----------
    response : callable
        Invalid signal conversion.

    Returns
    -------
    None
        Checks informative conversion errors.
    """
    with pytest.raises(ValueError, match="nonlinear_response"):
        detector_model(response=response).add_noise()


def test_callable_gain_requires_physical_energy():
    """Require an energy for calibration functions while allowing constants without one.

    Parameters
    ----------
    None
        Uses a low-level detector without a source-energy attribute.

    Returns
    -------
    None
        Checks energy validation and backward-compatible constant gain.
    """
    detector_model(5.0, energy=None).add_noise()
    with pytest.raises(ValueError, match="physical photon energy"):
        detector_model(energy_gain, energy=None).add_noise()


def experiment_config(path):
    """Declare a small integrating detector for scan and harmonic tests.

    Parameters
    ----------
    path : Path
        HDF5 destination.

    Returns
    -------
    ExperimentConfig
        Vacuum transmission with an energy-dependent gain and nonlinear camera.
    """
    return magnon_experiment().with_changes(
        xray={"photon_flux": 1e6},
        simulation={"shape": (8, 10), "real_space_pixel_size": 10e-9},
        sample={"recipe": "vacuum(2)", "max_slice_thickness": 2e-9},
        illumination={
            "energies_eV": (778.0, 790.0),
            "polarizations": ("LH",),
            "spectral_components": (
                sim.SpectralComponentConfig(weight=9.0),
                sim.SpectralComponentConfig(energy_factor=2.0, weight=1.0),
            ),
        },
        detector={
            "shape": (6, 8),
            "detector_center": (3, 4),
            "analyzer_angle": None,
            "projection_energy": 800.0,
            "detector_params": {
                "counts_per_photon": energy_gain,
                "nonlinear_response": compress_signal,
                "readout_noise_sigma": 0.0,
                "readout_noise_average": 0.0,
                "detector_threshold": 1e12,
                "noise_seed": 10,
            },
        },
        outputs={
            "path": path,
            "save": ("detector_ideal", "detector_measured"),
            "overwrite": True,
        },
    )


def test_hyperspectral_harmonics_use_real_energies_and_record_resolved_gain(tmp_path):
    """Check per-color gain through the standard runner with a coordinate-energy override.

    Parameters
    ----------
    tmp_path : Path
        Isolated output folder.

    Returns
    -------
    None
        Checks evaluated gain, callback provenance and seeded repeatability.
    """
    config = experiment_config(tmp_path / "response.h5")
    results = sim.simulate_experiment(config)
    measured = results.stack("detector_measured").copy()
    with h5py.File(results.path) as saved:
        for index, energy in enumerate((778.0, 790.0)):
            response = saved[f"runs/{index:06d}/detector_response"]
            np.testing.assert_array_equal(
                response["energies_eV"][()], [energy, 2 * energy]
            )
            np.testing.assert_allclose(
                response["counts_per_photon"][()], [energy / 100.0, energy / 50.0]
            )
            assert (
                "compress_signal"
                in json.loads(response.attrs["nonlinear_response_json"])["callback"]
            )
    repeated = sim.simulate_experiment(config)
    np.testing.assert_array_equal(measured, repeated.stack("detector_measured"))


def test_identity_response_and_default_linear_model_agree_for_multiple_frames():
    """Make an identity callback equivalent to the default, including saturation.

    Parameters
    ----------
    None
        Uses identical Poisson seeds and threshold settings in several frames.

    Returns
    -------
    None
        Checks that selecting a callback does not change frame integration.
    """
    linear = detector_model(5.0, frames=3)
    identity = detector_model(5.0, response=lambda signal: signal, frames=3)
    linear.detector_threshold = identity.detector_threshold = 480.0
    linear.add_noise()
    identity.add_noise()
    np.testing.assert_array_equal(linear.hologram_exp, identity.hologram_exp)
    assert linear.cosmic_ray_count == 0


def test_accumulated_response_uses_average_frame_and_spreads_once():
    from unittest import mock
    model = detector_model(2., lambda signal: signal**2, frames=50, rate=4.)
    photons = np.random.default_rng(123).poisson(np.full((12, 14), 200.))
    expected = np.round((2 * photons / 50)**2 * 50) / 50
    with mock.patch.object(model, "_spread_photons", wraps=model._spread_photons) as spread:
        model.add_noise(store_no_beamstop=True)
    assert spread.call_count == 1
    np.testing.assert_array_equal(model.hologram_exp, expected)
    np.testing.assert_array_equal(model.hologram_exp_no_beamstop, expected)


def test_accumulated_linear_noise_has_frame_average_variance():
    model = detector_model(2., frames=25, rate=40., shape=(400, 400))
    model.add_noise()
    assert abs(model.hologram_exp.mean() - 80.) < .05
    assert abs(model.hologram_exp.var() - 4 * 40 / 25) < .1


def test_acquisition_mode_is_validated_and_configurable():
    model = detector_model(frames=2)
    model._apply_measurement_config({"acquisition_mode": "per_frame", "number_frames": 2})
    assert model.acquisition_mode == "per_frame"
    model.acquisition_mode = "unknown"
    with pytest.raises(ValueError, match="acquisition_mode"):
        model.add_noise()
