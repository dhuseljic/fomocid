"""Tests for the hologram pipeline HDF5 configuration layout."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py

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
        assert sorted(h5["_pipeline_config"].keys()) == ["n_samples", "oversampling"]


def test_metadata_hierarchy_groups_sample_and_propagator_settings(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    cfg = pipeline.config
    aperture_config = {
        "aperture_types": cfg.aperture_types,
        "aperture_radii": cfg.aperture_radii,
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
        "sample/dielectric_tensor/compact": True,
        "propagator_config/propagator_method": "Jones",
        "propagator_config/propagate": False,
    }

    with h5py.File(tmp_path / "metadata.h5", "w") as h5:
        sample = h5.create_group("00000")
        pipeline._write_metadata(sample, metadata, aperture_config)

        assert "00000/metadata/sample/use_roi" in h5
        assert "00000/metadata/sample/aperture/aperture_config" in h5
        assert "00000/metadata/sample/dielectric_tensor/compact" in h5
        assert "00000/metadata/propagator_config/propagator_method" in h5
        assert "00000/metadata/propagator_config/propagate" in h5
        assert list(h5["00000/metadata/propagator_config"].keys())[0] == (
            "propagator_method"
        )
        assert "00000/metadata/use_roi" not in h5
        assert "00000/metadata/aperture" not in h5
        assert "00000/metadata/dielectric_tensor" not in h5
        assert "00000/metadata/propagation" not in h5
        assert "00000/metadata/propagator" not in h5


def test_sample_detector_metadata_uses_sibling_config_groups() -> None:
    detector_config = DetectorConfig()
    metadata = HologramPipeline._detector_metadata(detector_config)

    assert "measurement_config/exposure_time" in metadata
    assert "detector_params/counts_per_photon" in metadata
    assert "artifacts_config/sigma_photon" in metadata
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
