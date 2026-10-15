"""Dataset-driven nonmagnetic optical channels through the standard material path."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
from scipy.linalg import expm, sqrtm

from scattering_calculator import simulation_pipelines as sim
from scattering_calculator.database.database_loading import material_params
from scattering_calculator.simulation_pipelines.examples import (
    magnon_experiment,
    fth_experiment,
)


def write_channel(root, kind, values, energies=(776.0, 778.0, 780.0)):
    """Write synthetic calibration data only inside an isolated test dataset.

    Parameters
    ----------
    root : Path
        Test material database directory.
    kind : str
        Named material response channel.
    values : sequence of complex
        Index increments at the specified energies.
    energies : sequence of float
        Photon energy grid in eV.

    Returns
    -------
    Path
        Written energy/delta/beta dataset.
    """
    path = root / "Co" / f"Co_{kind}.txt"
    values = np.asarray(values, dtype=complex)
    np.savetxt(
        path,
        np.column_stack((energies, -values.real, -values.imag)),
        header="dimensionless optical constants\nenergy_eV delta beta",
        comments="",
    )
    return path


@pytest.fixture
def optical_database(tmp_path):
    """Provide test-only future material datasets beside existing Co constants.

    Parameters
    ----------
    tmp_path : Path
        Isolated test directory.

    Returns
    -------
    Path
        Dataset root; no synthetic coefficients enter the packaged database.
    """
    bundled = (
        Path(__file__).parents[1]
        / "src/scattering_calculator/database/material_parameter/refractive_indexes"
    )
    root = tmp_path / "materials"
    shutil.copytree(bundled / "Co", root / "Co")
    for kind in ("natural_linear", "ferroelectric"):
        write_channel(root, kind, [2e-3 - 1e-3j] * 3)
    write_channel(root, "natural_circular", [1e-4 - 1e-4j, 2e-4 - 2e-4j, 3e-4 - 3e-4j])
    return root


def configuration(database=None, responses=(), mode="Jones"):
    """Declare one small film using the standard material loader.

    Parameters
    ----------
    database : Path or None
        Refractive-index dataset root.
    responses : tuple
        Spatial selectors without inline optical coefficients.
    mode : str
        Field representation.

    Returns
    -------
    ExperimentConfig
        Uniform film configuration.
    """
    return magnon_experiment().with_changes(
        xray={"pol": "LH", "photon_flux": 10000.0, "linear_polarization_angle": None},
        simulation={"shape": (8, 10), "real_space_pixel_size": 10e-9},
        sample={
            "recipe": "Co(4)",
            "max_slice_thickness": 2e-9,
            "refractive_index_database": database,
            "optical_responses": responses,
        },
        propagation={
            "propagator_method": mode,
            "propagator_config": {"propagate": False},
        },
        detector={"shape": (6, 8), "detector_center": (3, 4), "analyzer_angle": None},
    )


def prepared(config):
    """Prepare and propagate a film without magnetization.

    Parameters
    ----------
    config : ExperimentConfig
        Film declaration.

    Returns
    -------
    ScatteringExperiment
        Propagated sample.
    """
    run = sim.ScatteringExperiment(config).setup(
        magnetization=np.zeros((*config.simulation.shape, 3))
    )
    run.propagate()
    return run


def test_defaults_need_no_new_datasets_and_inline_spectra_are_not_accepted():
    """Keep the extension off by default and require material-owned optical data.

    Parameters
    ----------
    None
        Uses only the existing bundled dataset.

    Returns
    -------
    None
        Checks default compatibility and missing-channel diagnostics.
    """
    assert sim.SampleConfig().optical_responses == ()
    first = prepared(configuration())
    second = prepared(configuration(responses=[]))
    np.testing.assert_array_equal(first.wavefront.exit_wave, second.wavefront.exit_wave)
    with pytest.raises(TypeError):
        sim.OpticalResponseConfig("Co", "ferroelectric", delta_n=1e-3j)
    with pytest.raises(
        FileNotFoundError, match="ferroelectric refractive-index dataset"
    ):
        prepared(
            configuration(responses=(sim.OpticalResponseConfig("Co", "ferroelectric"),))
        )


@pytest.mark.parametrize("kind", ["natural_linear", "ferroelectric"])
def test_dataset_axis_response_matches_independent_matrix_transmission(
    optical_database, kind
):
    """Check the material coefficient, conversion and axis-reversal symmetry.

    Parameters
    ----------
    optical_database : Path
        Test-only refractive-index dataset root.
    kind : str
        Linear response channel.

    Returns
    -------
    None
        Compares exit fields against an independent exponential.
    """
    response = sim.OpticalResponseConfig("Co", kind, axis=(1.0, 1.0, 0.0))
    run = prepared(configuration(optical_database, (response,)))
    n0 = run.sample.layer_refractive_indices[0][0]
    increment = run.sample.material_params.get_optical_response("Co", kind)
    assert increment == 2e-3 - 1e-3j
    axis = np.array([1.0, 1.0]) / np.sqrt(2.0)
    epsilon = n0**2 * np.eye(2) + 2 * n0 * increment * np.outer(axis, axis)
    k = 2 * np.pi / run.config.illumination.beam_params.wavelength
    transmission = expm(-1j * k * 4e-9 * np.asarray(sqrtm(epsilon), dtype=complex))
    entrance = run.illumination.illumination.illumination_jones
    expected = np.einsum("ij,...j->...i", transmission, entrance) * np.exp(
        -1j * k * 2e-9
    )
    np.testing.assert_allclose(
        run.wavefront.exit_wave, expected, rtol=1e-11, atol=1e-12
    )
    reverse = prepared(
        configuration(optical_database, (replace(response, axis=(-1.0, -1.0, 0.0)),))
    )
    np.testing.assert_allclose(run.wavefront.exit_wave, reverse.wavefront.exit_wave)
    assert np.linalg.norm(run.wavefront.exit_wave[..., 1]) > 0


def test_natural_circular_handedness_reciprocity_and_scalar_eigenmode(optical_database):
    """Propagate the same dataset through circular polarization and handedness changes.

    Parameters
    ----------
    optical_database : Path
        Test-only response dataset root.

    Returns
    -------
    None
        Verifies helicity contrast, scalar agreement and reciprocal lab tensor.
    """
    response = sim.OpticalResponseConfig("Co", "natural_circular", axis=None)
    config = configuration(optical_database, (response,)).with_changes(
        xray={"pol": "CR"}
    )
    right = prepared(config)
    left = prepared(config.with_changes(xray={"pol": "CL"}))
    reverse = prepared(
        config.with_changes(
            sample={"optical_responses": (replace(response, strength=-1.0),)}
        )
    )
    assert not np.allclose(
        abs(right.wavefront.exit_wave) ** 2, abs(left.wavefront.exit_wave) ** 2
    )
    np.testing.assert_allclose(
        abs(left.wavefront.exit_wave) ** 2, abs(reverse.wavefront.exit_wave) ** 2
    )
    scalar = prepared(config.with_changes(propagation={"propagator_method": "Scalar"}))
    np.testing.assert_allclose(
        abs(scalar.wavefront.exit_wave) ** 2,
        np.sum(abs(right.wavefront.exit_wave) ** 2, axis=-1),
    )
    right.sample.calculate_final_dielectric_tensor(local_k_projection=True)
    stack = right.sample.final_dielectric_tensor
    forward = stack.project_layer(0, np.broadcast_to([0.0, 0.0, 1.0], (8, 10, 3)))
    backward = stack.project_layer(0, np.broadcast_to([0.0, 0.0, -1.0], (8, 10, 3)))
    basis = np.diag([1.0, -1.0])
    np.testing.assert_allclose(basis @ backward @ basis, forward.swapaxes(-1, -2))


def test_dataset_response_respects_vacuum_and_nominal_local_projection(
    optical_database,
):
    """Keep the material response out of vacuum and project local axes consistently.

    Parameters
    ----------
    optical_database : Path
        Test-only datasets.

    Returns
    -------
    None
        Compares nominal/local tensors and Jones/Stokes detector intensities.
    """
    axis = np.zeros((8, 10, 3))
    axis[..., 0] = 1
    axis[:, 5:, 0] = -1
    response = sim.OpticalResponseConfig("Co", "ferroelectric", axis=axis)
    run = prepared(configuration(optical_database, (response,)))
    run.sample.mask[:, :, :2] = 0
    run.sample.calculate_final_dielectric_tensor(local_k_projection=True)
    eps = run.sample.final_dielectric_tensor.project_layer(0)
    np.testing.assert_allclose(eps[:, :2], np.broadcast_to(np.eye(2), (8, 2, 2, 2)))
    run.sample.calculate_final_dielectric_tensor(compact=True)
    np.testing.assert_allclose(eps, run.sample.final_dielectric_tensor[0])
    jones = prepared(configuration(optical_database, (response,)))
    stokes = prepared(configuration(optical_database, (response,), mode="Stokes"))
    np.testing.assert_allclose(
        jones.propagate_to_detector(), stokes.propagate_to_detector()
    )


def test_dataset_energy_interpolation_and_recorded_provenance(
    optical_database, tmp_path
):
    """Refresh material channels with each energy and save their evaluated data.

    Parameters
    ----------
    optical_database : Path
        Isolated material datasets.
    tmp_path : Path
        HDF5 output directory.

    Returns
    -------
    None
        Checks interpolation, range errors, hashes and scan axes.
    """
    value = material_params.load_optical_response(
        "Co", "natural_circular", 777.0, optical_database
    )
    assert np.isclose(value, 1.5e-4 - 1.5e-4j)
    with pytest.raises(ValueError, match="outside"):
        material_params.load_optical_response(
            "Co", "natural_circular", 781.0, optical_database
        )
    response = sim.OpticalResponseConfig("Co", "natural_circular", axis=None)
    config = configuration(optical_database, [response]).with_changes(
        illumination={"energies_eV": (777.0, 779.0), "polarizations": ("LH",)},
        outputs={
            "path": tmp_path / "optical.h5",
            "save": ("exit_wave", "detector_ideal"),
        },
    )
    results = sim.simulate_experiment(config)
    assert results.stack("detector_ideal").shape == (2, 1, 6, 8)
    assert results.parameter_axes == {}
    with h5py.File(results.path) as saved:
        record = json.loads(saved["config_json"][()])
        assert "spectrum" not in record["sample"]["optical_responses"][0]
        first = saved["runs/000000/materials/optical_responses/000000"]
        second = saved["runs/000001/materials/optical_responses/000000"]
        assert len(first.attrs["sha256"]) == 64
        assert np.isclose(first["index_increment"][()], value)
        assert not np.isclose(second["index_increment"][()], value)


def test_named_magnetic_linear_channel_uses_the_same_loader(optical_database):
    """Populate the existing XMLD slot from a named material table.

    Parameters
    ----------
    optical_database : Path
        Test-only material root.

    Returns
    -------
    None
        Checks that magnetic and nonmagnetic channels share lookup conventions.
    """
    write_channel(optical_database, "magnetic_linear", [2e-4 - 3e-4j] * 3)
    indices = material_params.load_refractive_index("Co", 778.0, optical_database)
    assert indices[2] == 2e-4 - 3e-4j
    database = material_params(
        materials=["Co"], x_ray_energy=778.0, db_path=optical_database
    )
    assert database.get_optical_response("Co", "magnetic_linear") == indices[2]


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "unknown"},
        {"axis": (0.0, 0.0, 0.0)},
        {"strength": np.inf},
        {"axis": np.ones((3, 4))},
    ],
)
def test_invalid_spatial_order_is_rejected(optical_database, change):
    """Validate spatial fields separately from material spectra.

    Parameters
    ----------
    optical_database : Path
        Test-only datasets.
    change : dict
        Invalid spatial selector override.

    Returns
    -------
    None
        Checks informative failures.
    """
    response = replace(sim.OpticalResponseConfig("Co", "ferroelectric"), **change)
    with pytest.raises(ValueError):
        prepared(configuration(optical_database, (response,)))


def test_scalar_mixing_and_absent_material_fail(optical_database):
    """Reject polarization mixing in scalar mode and unavailable sample materials.

    Parameters
    ----------
    optical_database : Path
        Test datasets.

    Returns
    -------
    None
        Checks explicit errors rather than silently ignoring requested effects.
    """
    response = sim.OpticalResponseConfig("Co", "natural_linear", axis=(1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="Jones or Stokes"):
        prepared(configuration(optical_database, (response,), mode="Scalar"))
    with pytest.raises(ValueError, match="absent"):
        prepared(configuration(optical_database, (replace(response, material="Au"),)))


def test_mixed_stokes_retains_spatial_phase_with_anisotropic_padding(optical_database):
    """Compare mixed-polarization propagation with independent Jones carriers.

    Parameters
    ----------
    optical_database : Path
        Test material root.

    Returns
    -------
    None
        Checks tilt, defocus, sample response and background transmission.
    """
    response = sim.OpticalResponseConfig("Co", "ferroelectric", axis=(1.0, 1.0, 0.0))
    config = configuration(optical_database, (response,)).with_changes(
        illumination={
            "illumination_config": {
                "center": (0.0, 0.0),
                "fwhm": 80e-9,
                "distance": 2e-6,
                "alpha_beam": (0.0, 0.01),
            }
        },
        propagation={
            "propagator_method": "Stokes",
            "propagator_config": {
                "propagate": True,
                "input_stokes": (1.0, 0.4, 0.0, 0.0),
                "farfield_oversampling": 2,
            },
        },
    )
    mixture = prepared(config)
    pure = config.with_changes(
        propagation={
            "propagator_method": "Jones",
            "propagator_config": {"propagate": True, "farfield_oversampling": 2},
        }
    )
    horizontal = prepared(pure.with_changes(xray={"pol": "LH"}))
    vertical = prepared(pure.with_changes(xray={"pol": "LV"}))
    np.testing.assert_allclose(
        mixture.propagate_to_detector(),
        0.7 * horizontal.propagate_to_detector()
        + 0.3 * vertical.propagate_to_detector(),
        rtol=1e-11,
        atol=1e-12,
    )


def test_zero_material_channel_preserves_existing_fth_support(optical_database):
    """Check zero added dataset coefficients on the existing masked FTH geometry.

    Parameters
    ----------
    optical_database : Path
        Test root containing a zero-valued ferroelectric channel.

    Returns
    -------
    None
        Compares the original and opt-in exit fields.
    """
    write_channel(optical_database, "ferroelectric", [0j] * 3)
    config = fth_experiment().with_changes(
        simulation={"shape": (64, 64), "real_space_pixel_size": 10e-9},
        sample={
            "recipe": "Co(4)",
            "max_slice_thickness": 2e-9,
            "refractive_index_database": optical_database,
        },
        aperture={
            "aperture_config": {
                "apertures_type": ["OH", "RH"],
                "apertures_radius": [50e-9, 15e-9],
                "apertures_center": [(0.0, 0.0), (0.0, 200e-9)],
                "thickness_OH": 0.0,
            }
        },
    )
    original = sim.ScatteringExperiment(config).setup()
    original.propagate()
    response = sim.OpticalResponseConfig("Co", "ferroelectric")
    extended = sim.ScatteringExperiment(
        config.with_changes(sample={"optical_responses": (response,)})
    ).setup(magnetization=original.sample.magnetization, mask=original.sample.mask)
    extended.propagate()
    np.testing.assert_allclose(
        original.wavefront.exit_wave,
        extended.wavefront.exit_wave,
        rtol=1e-10,
        atol=1e-12,
    )


def test_simultaneous_modes_reload_material_data_at_each_physical_energy(
    optical_database, tmp_path
):
    """Combine dataset-driven optical response, polarization modes and detector callbacks.

    Parameters
    ----------
    optical_database : Path
        Isolated material datasets with energy-dependent natural circular response.
    tmp_path : Path
        Result directory.

    Returns
    -------
    None
        Checks per-component channel values and detector calibration metadata.
    """
    response = sim.OpticalResponseConfig("Co", "natural_circular", axis=None)
    config = configuration(optical_database, (response,), mode="Stokes").with_changes(
        propagation={
            "propagator_config": {
                "propagate": True,
                "input_stokes": (1.0, 0.4, 0.0, 0.0),
            }
        },
        illumination={
            "energies_eV": (778.0,),
            "polarizations": ("LH",),
            "spectral_components": (
                sim.SpectralComponentConfig(energy=777.0, weight=2.0),
                sim.SpectralComponentConfig(energy=779.0, weight=3.0),
            ),
        },
        detector={
            "detector_params": {
                "counts_per_photon": lambda energy: energy / 100.0,
                "nonlinear_response": lambda signal: signal / (1.0 + signal / 1000.0),
                "readout_noise_sigma": 0.0,
                "readout_noise_average": 0.0,
                "noise_seed": 5,
            }
        },
        outputs={
            "path": tmp_path / "combined.h5",
            "save": ("detector_ideal", "detector_measured"),
        },
    )
    run = prepared(config)
    expected = np.zeros(config.detector.shape)
    for energy, weight in ((777.0, 0.4), (779.0, 0.6)):
        component = prepared(
            config.with_changes(
                xray={"energy": energy, "photon_flux": 10000.0 * weight},
                illumination={"spectral_components": None},
            )
        )
        expected += component.propagate_to_detector()
    np.testing.assert_allclose(
        run.propagate_to_detector(), expected, rtol=1e-10, atol=1e-12
    )
    result = sim.simulate_experiment(config, magnetization=np.zeros((8, 10, 3)))
    with h5py.File(result.path) as saved:
        base = "runs/000000/spectral_components"
        for index, energy in enumerate((777.0, 779.0)):
            data = saved[f"{base}/{index:06d}/optical_responses/000000"]
            assert data.attrs["energy_eV"] == energy
            assert np.isclose(
                data["index_increment"][()],
                material_params.load_optical_response(
                    "Co", "natural_circular", energy, optical_database
                ),
            )
        np.testing.assert_allclose(
            saved["runs/000000/detector_response/counts_per_photon"][()], [7.77, 7.79]
        )
