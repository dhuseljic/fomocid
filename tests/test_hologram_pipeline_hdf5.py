"""Tests for the hologram pipeline HDF5 configuration layout."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scattering_calculator.simulation_pipelines.pipelines.hologram_pipeline import (
    HologramPipeline,
    HologramPipelineConfig,
    HologramPipelineRanges,
)
from scattering_calculator.simulation_pipelines.simulation_configuration import (
    HologramConfig,
    XRayConfig,
)
from scattering_calculator.simulation_pipelines.simulation_configuration import (
    DetectorConfig,
    MagneticPatternConfig,
)


def _pipeline(tmp_path: Path, **config_changes) -> HologramPipeline:
    config = HologramPipelineConfig(recipe="SiN(80)", **config_changes)
    return HologramPipeline(
        config=config,
        ranges=HologramPipelineRanges(),
        output_path=tmp_path / "output.h5",
        n_samples=1,
        verbose=False,
    )


def test_pipeline_config_does_not_duplicate_per_sample_configs(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)

    with h5py.File(tmp_path / "config.h5", "w") as h5:
        pipeline._write_pipeline_config(h5)
        assert "_pipeline_config/measurement_config" not in h5
        assert "_pipeline_config/detector_params" not in h5
        assert "_pipeline_config/artifacts_config" not in h5
        assert "_pipeline_config/beamstop_config" not in h5
        assert "_pipeline_config/detector_quantum_efficiency" not in h5
        assert "_pipeline_config/recipe" not in h5
        assert "_pipeline_config/propagate" not in h5
        assert "_pipeline_config/propagator_method" not in h5
        assert "_pipeline_config/propagator_config/propagator_method" not in h5
        assert sorted(h5["_pipeline_config"].keys()) == ["n_samples", "oversampling"]


def test_detector_center_can_be_sampled_from_ranges(tmp_path: Path) -> None:
    pipeline = HologramPipeline(
        config=HologramPipelineConfig(
            recipe="SiN(80)",
            detector_shape=(100, 120),
            detector_center=(50, 60),
        ),
        ranges=HologramPipelineRanges(
            detector_center=lambda params: (
                params["detector_shape"][0] / 2 + 7,
                params["detector_shape"][1] / 2 - 5,
            )
        ),
        output_path=tmp_path / "output.h5",
        n_samples=1,
        verbose=False,
    )

    params = pipeline._sample_params()

    assert params["detector_center"] == (57, 55)


def test_hologram_config_supports_linear_polarization_differences() -> None:
    cfg = HologramConfig(
        ideal_holograms={
            "LH": np.full((2, 2), 3.0),
            "LV": np.full((2, 2), 1.0),
        },
        detected_holograms={
            "LH": np.full((2, 2), 4.0),
            "LV": np.full((2, 2), 2.0),
        },
        exit_waves={
            "LH": np.full((2, 2), 5.0 + 1.0j),
            "LV": np.full((2, 2), 1.0 + 0.5j),
        },
    )

    cfg.compute_differences("LH", "LV", key="linear_diff")
    cfg.compute_sums("LH", "LV", key="linear_sum")

    np.testing.assert_allclose(cfg.ideal_holograms["linear_diff"], 2.0)
    np.testing.assert_allclose(cfg.detected_holograms["linear_sum"], 6.0)
    np.testing.assert_allclose(cfg.exit_waves["linear_diff"], 4.0 + 0.5j)


def test_xray_config_accepts_preferred_linear_polarization_labels() -> None:
    horizontal = XRayConfig(energy=778.0, photon_flux=1e8, pol="LH")
    vertical = XRayConfig(energy=778.0, photon_flux=1e8, pol="LV")

    assert horizontal.setup().pol == "LH"
    assert vertical.setup().pol == "LV"


def test_write_precomputed_result_requires_single_sample(tmp_path: Path) -> None:
    pipeline = HologramPipeline(
        config=HologramPipelineConfig(recipe="SiN(80)"),
        ranges=HologramPipelineRanges(),
        output_path=tmp_path / "output.h5",
        n_samples=2,
        verbose=False,
    )

    try:
        pipeline.write_precomputed_result(None, None, {}, {}, None, None)
    except ValueError as error:
        assert "exactly one sample" in str(error)
    else:
        raise AssertionError("Expected multi-sample precomputed export to fail")


def test_write_precomputed_result_uses_pipeline_hdf5_layout(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    cfg = pipeline.config
    aperture_config = {
        "aperture_types": cfg.aperture_types,
        "aperture_radii": cfg.aperture_radii,
        "aperture_lengths": cfg.aperture_lengths,
        "aperture_centers": cfg.aperture_centers,
        "aperture_sigmas": cfg.aperture_sigmas,
        "aperture_angles": cfg.aperture_angles,
        "aperture_ellipticities": cfg.aperture_ellipticities,
        "aperture_roughnesses": cfg.aperture_roughnesses,
        "aperture_roughness_modes": cfg.aperture_roughness_modes,
        "aperture_seeds": cfg.aperture_seeds,
        "aperture_top_radius_factors": cfg.aperture_top_radius_factors,
    }
    hologram = SimpleNamespace(
        to_dict=lambda **_: {
            "CR": {"exit_wave": np.ones((4, 4), dtype=complex), "ideal": np.ones((4, 4))},
            "CL": {"exit_wave": np.ones((4, 4), dtype=complex), "ideal": np.ones((4, 4))},
        }
    )
    detector = SimpleNamespace(
        detector_layout=SimpleNamespace(beamstop=np.ones((4, 4)))
    )

    written = pipeline.write_precomputed_result(
        hologram_config=hologram,
        detector_config=detector,
        metadata={"sample/recipe": "SiN(80)"},
        aperture_config=aperture_config,
        supportmask=np.ones((4, 4)),
        magnetic_pattern_oh=np.ones((4, 4)),
        overwrite=True,
    )

    with h5py.File(written, "r") as h5:
        assert "_pipeline_config/oversampling" in h5
        assert "00000/CR/exit_wave" in h5
        assert "00000/CL/ideal" in h5
        assert "00000/supportmask" in h5
        assert "00000/magnetic_pattern_oh" in h5
        assert "00000/metadata/sample/recipe" in h5
        assert "00000/metadata/sample/aperture/aperture_config" in h5
        assert "00000/metadata/sample/aperture/aperture_config/apertures_length" in h5


def test_build_precomputed_metadata_uses_canonical_groups() -> None:
    def config_with_metadata(values):
        return SimpleNamespace(
            get_metadata=lambda prefix="": {
                f"{prefix}{key}": value for key, value in values.items()
            }
        )

    xray = SimpleNamespace(
        beam_params=SimpleNamespace(
            energy=778.0,
            photon_flux=1e5,
            coherence_length=(1e-6, 1e-6),
            wavelength=1e-9,
            wavevector=2 * np.pi / 1e-9,
        )
    )
    detector = config_with_metadata(
        {
            "shape": (4, 4),
            "detector_params/counts_per_photon": 100,
            "measurement_config/exposure_time": 1.0,
        }
    )
    magnetic = config_with_metadata(
        {"pattern_type_method": "saturated_pattern", "real_space_pixel_size": 1e-9}
    )
    aperture = config_with_metadata(
        {"aperture_method": "FTH_circular", "real_space_pixel_size": 1e-9}
    )
    metadata = HologramPipeline.build_precomputed_metadata(
        xray_config=xray,
        detector_config=detector,
        beamstop_config=config_with_metadata({"bs_method": "circular"}),
        sample_config=config_with_metadata({"recipe": "SiN(80)"}),
        magnetic_pattern_config=magnetic,
        aperture_config=aperture,
        illumination_config=SimpleNamespace(
            illumination_function="gaussian",
            illumination_config={
                "center": (0.0, 0.0),
                "distance": 1e-3,
                "fwhm": 1e-6,
                "alpha_beam": (0.1, 0.2),
            },
        ),
        propagator_config=SimpleNamespace(
            propagator_method="Jones",
            propagator_config={"propagate": False},
        ),
        use_roi=True,
        magnetic_pattern_use_roi=False,
        dielectric_tensor_use_roi=True,
        dielectric_tensor_compact=True,
        save_detected_hologram_without_beamstop=False,
    )

    assert "sample/magnetic_pattern/pattern_type_method" in metadata
    assert "sample/aperture/aperture_method" in metadata
    assert "sample/dielectric_tensor/compact" in metadata
    assert "propagator_config/propagator_method" in metadata
    assert "propagator_method" not in metadata
    assert "detector_params/counts_per_photon" in metadata
    assert "measurement_config/exposure_time" in metadata
    assert "detector/detector_params/counts_per_photon" not in metadata
    assert np.array_equal(metadata["illumination/alpha_beam_rad"], (0.1, 0.2))


def test_metadata_hierarchy_groups_sample_and_propagator_settings(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    cfg = pipeline.config
    aperture_config = {
        "aperture_types": cfg.aperture_types,
        "aperture_radii": cfg.aperture_radii,
        "aperture_lengths": cfg.aperture_lengths,
        "aperture_centers": cfg.aperture_centers,
        "aperture_sigmas": cfg.aperture_sigmas,
        "aperture_angles": cfg.aperture_angles,
        "aperture_ellipticities": cfg.aperture_ellipticities,
        "aperture_roughnesses": cfg.aperture_roughnesses,
        "aperture_roughness_modes": cfg.aperture_roughness_modes,
        "aperture_seeds": cfg.aperture_seeds,
        "aperture_top_radius_factors": cfg.aperture_top_radius_factors,
    }
    metadata = {
        "sample/use_roi": True,
        "sample/magnetic_pattern/use_roi": True,
        "sample/dielectric_tensor/compact": True,
        "propagator_config/propagate": False,
        "propagator_config/jones_apply_zero_order_phase": True,
        "propagator_config/propagator_method": "Jones",
    }

    with h5py.File(tmp_path / "metadata.h5", "w") as h5:
        sample = h5.create_group("00000")
        pipeline._write_metadata(sample, metadata, aperture_config)

        assert "00000/metadata/sample/use_roi" in h5
        assert "00000/metadata/sample/magnetic_pattern/use_roi" in h5
        assert "00000/metadata/sample/aperture/aperture_config" in h5
        assert "00000/metadata/sample/dielectric_tensor/compact" in h5
        assert "00000/metadata/propagator_config/propagator_method" in h5
        assert "00000/metadata/propagator_config/propagate" in h5
        assert "00000/metadata/propagator_config/jones_apply_zero_order_phase" in h5
        assert list(h5["00000/metadata/propagator_config"].keys())[0] == (
            "propagator_method"
        )
        assert "00000/metadata/propagator_method" not in h5
        assert "00000/metadata/use_roi" not in h5
        assert "00000/metadata/aperture" not in h5
        assert "00000/metadata/dielectric_tensor" not in h5
        assert "00000/metadata/magnetic_pattern" not in h5
        assert "00000/metadata/propagation" not in h5
        assert "00000/metadata/propagator" not in h5


def test_sample_detector_metadata_uses_sibling_config_groups() -> None:
    detector_config = DetectorConfig(
        use_detector_pixel_footprint=True,
        detector_pixel_footprint_samples=5,
    )
    metadata = HologramPipeline._detector_metadata(detector_config)

    assert "measurement_config/exposure_time" in metadata
    assert "detector_params/counts_per_photon" in metadata
    assert "artifacts_config/sigma_photon" in metadata
    assert metadata["detector/use_detector_pixel_footprint"] is True
    assert metadata["detector/detector_pixel_footprint_samples"] == 5
    assert "detector/measurement_config/exposure_time" not in metadata
    assert "detector/artifacts_config/sigma_photon" not in metadata
    assert "artifacts_config/counts_per_photon" not in metadata


def test_pipeline_rejects_conflicting_counts_per_photon(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        detector_params={"counts_per_photon": 180},
        artifacts_config={"counts_per_photon": 100},
    )

    try:
        pipeline._sample_params()
    except ValueError as error:
        assert "Conflicting counts_per_photon" in str(error)
    else:
        raise AssertionError("Expected conflicting counts_per_photon values to fail")


def test_pipeline_rejects_conflicting_legacy_detector_fields(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        detector_noise_rms=2,
        detector_quantum_efficiency=0.8,
        detector_params={
            "readout_noise_sigma": 3,
            "quantum_efficiency": 0.9,
        },
    )

    try:
        pipeline._sample_params()
    except ValueError as error:
        assert "Conflicting detector_" in str(error)
    else:
        raise AssertionError("Expected conflicting legacy detector fields to fail")


def test_legacy_detector_fields_are_normalized_when_canonical_values_absent(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path,
        detector_noise_rms=2,
        detector_quantum_efficiency=0.8,
        detector_params={"counts_per_photon": 100},
    )

    params = pipeline._sample_params()["detector_params"]
    assert params["readout_noise_sigma"] == 2
    assert params["quantum_efficiency"] == 0.8
    assert "noise_rms" not in params


def test_detector_pixel_footprint_config_is_sampled(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        use_detector_pixel_footprint=True,
        detector_pixel_footprint_samples=5,
    )

    params = pipeline._sample_params()
    assert params["use_detector_pixel_footprint"] is True
    assert params["detector_pixel_footprint_samples"] == 5


def test_illumination_alpha_beam_config_is_sampled(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        illumination_alpha_beam=(0.1, 0.15),
    )

    params = pipeline._sample_params()
    assert params["illumination_alpha_beam"] == (0.1, 0.15)


def test_xray_metadata_has_one_canonical_group() -> None:
    from scattering_calculator.simulation_pipelines.simulation_configuration import (
        XRayConfig,
    )

    xray = XRayConfig(energy=778.0, photon_flux=1e5)
    xray.setup()
    metadata = HologramPipeline._xray_metadata(xray)

    assert sorted(metadata) == [
        "xray/coherence_length_m",
        "xray/energy_eV",
        "xray/photon_flux",
        "xray/wavelength_m",
        "xray/wavevector_per_m",
    ]


def test_pipeline_rejects_conflicting_legacy_pattern_config(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        pattern_config={"stripe_width": 50e-9},
        pattern_config_length={"stripe_width": 70e-9},
    )

    try:
        pipeline._sample_params()
    except ValueError as error:
        assert "pattern_config_length is a legacy alias" in str(error)
    else:
        raise AssertionError("Expected conflicting pattern configuration to fail")


def test_magnetic_pattern_config_rejects_conflicting_legacy_lengths() -> None:
    config = MagneticPatternConfig(
        pattern_type_method="saturated_pattern",
        shape=(8, 8),
        real_space_pixel_size=1e-9,
        pattern_config={"saturation": 1.0},
        pattern_config_length={"saturation": -1.0},
    )

    try:
        config.create_pattern()
    except ValueError as error:
        assert "pattern_config_length is a legacy alias" in str(error)
    else:
        raise AssertionError("Expected conflicting pattern configuration to fail")


def test_skyrmion_sigma_is_converted_from_metres_and_softens_edges() -> None:
    common = {
        "pattern_type_method": "skyrmion_pattern",
        "shape": (48, 48),
        "real_space_pixel_size": 2e-9,
        "pattern_config": {
            "skyr_radius": 8e-9,
            "screening_radius": 10e-9,
            "number_skyr": 1,
            "number_iter": 1,
            "seed": 3,
        },
    }
    sharp = MagneticPatternConfig(**common)
    smooth = MagneticPatternConfig(
        **{
            **common,
            "pattern_config": {
                **common["pattern_config"],
                "sigma": 4e-9,
            },
        }
    )

    sharp_pattern, _ = sharp.create_pattern()
    smooth_pattern, _ = smooth.create_pattern()

    sharp_intermediate = np.count_nonzero((sharp_pattern > -1.0) & (sharp_pattern < 1.0))
    smooth_intermediate = np.count_nonzero((smooth_pattern > -1.0) & (smooth_pattern < 1.0))

    assert sharp_intermediate > 0
    assert np.any((smooth_pattern > -1.0) & (smooth_pattern < 1.0))
    assert smooth_intermediate > sharp_intermediate


def test_disordered_skyrmion_sigma_softens_edges() -> None:
    config = MagneticPatternConfig(
        pattern_type_method="disordered_skyrmion_lattice_pattern",
        shape=(64, 64),
        real_space_pixel_size=2e-9,
        pattern_config={
            "stripe_width": 16e-9,
            "sigma": 4e-9,
            "skyrmion_density": 0.1,
            "ellipticity": (1.0, 1.0),
            "roughness": 0.0,
            "seed": 4,
        },
    )

    pattern, _ = config.create_pattern()

    assert np.any((pattern > -1.0) & (pattern < 1.0))
