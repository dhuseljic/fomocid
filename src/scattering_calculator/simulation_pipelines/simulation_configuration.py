from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from scattering_calculator.experimental_conditions import detector, light_beam
from scattering_calculator.sample_generator import pattern_generator
from scattering_calculator.sample_generator import structures
from scattering_calculator.beam_propagator import Jones_propagator


class _ConfigMixin:
    def to_dict(self) -> dict:
        """Return all configuration fields and their current values as a dict."""
        return dataclasses.asdict(self)


@dataclass
class XRayConfig(_ConfigMixin):
    energy: float  # eV
    photon_flux: float  # photons/s
    polarization: Literal["CR", "CL", "x", "y"] = "CR"
    coherence_length: float = 10e-6  # m

    def __post_init__(self) -> None:
        if self.energy <= 0:
            raise ValueError(f"energy must be positive, got {self.energy}")
        if self.photon_flux <= 0:
            raise ValueError(f"photon_flux must be positive, got {self.photon_flux}")
        if self.coherence_length <= 0:
            raise ValueError(
                f"coherence_length must be positive, got {self.coherence_length}"
            )
        if self.polarization not in ["CR", "CL", "x", "y"]:
            raise ValueError(
                f"Polarisation must be in CR, CL, x or, got {self.polarization}"
            )

    def setup(self) -> light_beam.beam_parameters:
        return light_beam.beam_parameters(
            self.energy, self.photon_flux, self.polarization, self.coherence_length
        )


@dataclass
class BeamstopConfig(_ConfigMixin):
    bs_method: Literal["circular", None] | None = "circular"
    bs_detector_distance: float = 0.01  # m
    bs_center: tuple[int, int] = (0, 0)  # px
    bs_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bs_detector_distance <= 0:
            raise ValueError(
                f"bs_detector_distance must be positive, got {self.bs_detector_distance}"
            )

    def setup(self, detector_layout: detector.detector_layout) -> detector.beamstop:
        bs = detector.beamstop(
            detector_config=detector_layout,
            distance_detector_beamstop=self.bs_detector_distance,
        )
        if self.bs_method is None:
            return bs.create_empty_beamstop()
        if self.bs_method == "circular":
            bs.create_circle_beamstop(center=self.bs_center, **self.bs_config)
        return bs


@dataclass
class DetectorConfig(_ConfigMixin):
    shape: tuple[int, int] = (256, 256)  # px
    pixel_size: float = 55e-6  # m/px
    sample_to_detector_distance: float = 0.1  # m
    detector_center: tuple[int, int] = (0, 0)  # px
    detector_efficiency: float = 1.0  # 0–1
    detector_noise_rms: float = 0.0  # counts
    artifacts_method: str | None = None
    artifacts_config: dict = field(default_factory=dict)
    beamstop: BeamstopConfig | None = None

    def __post_init__(self) -> None:
        if any(s <= 0 for s in self.shape):
            raise ValueError(f"shape dimensions must be positive, got {self.shape}")
        if self.pixel_size <= 0:
            raise ValueError(f"pixel_size must be positive, got {self.pixel_size}")
        if self.sample_to_detector_distance <= 0:
            raise ValueError(
                f"sample_to_detector_distance must be positive, got {self.sample_to_detector_distance}"
            )
        if not (0.0 <= self.detector_efficiency <= 1.0):
            raise ValueError(
                f"detector_efficiency must be in [0, 1], got {self.detector_efficiency}"
            )
        if self.detector_noise_rms < 0:
            raise ValueError(
                f"detector_noise_rms must be non-negative, got {self.detector_noise_rms}"
            )

    def setup(self):
        self.detector_layout = detector.detector_layout(
            pixel_size=self.pixel_size,
            detector_shape=self.shape,
            distance_sample_detector=self.sample_to_detector_distance,
            detector_center=self.detector_center,
        )
        if self.beamstop is not None:
            bs = self.beamstop.setup(self.detector_layout)
            self.detector_layout.assign_beamstop(bs.beamstop)
        return self.detector_layout


@dataclass
class SimulationConfig(_ConfigMixin):
    shape: tuple[int, int] = (256, 256)  # px
    real_space_pixel_size: float = 10e-9  # m/px
    other_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(s <= 0 for s in self.shape):
            raise ValueError(f"shape dimensions must be positive, got {self.shape}")
        if self.real_space_pixel_size <= 0:
            raise ValueError(
                f"real_space_pixel_size must be positive, got {self.real_space_pixel_size}"
            )

    def setup(self) -> dict:
        y = (np.arange(self.shape[0]) - self.shape[0] / 2) * self.real_space_pixel_size
        x = (np.arange(self.shape[1]) - self.shape[1] / 2) * self.real_space_pixel_size
        X, Y = np.meshgrid(x, y)
        return {
            "shape": self.shape,
            "pixel_size": self.real_space_pixel_size,
            "x": X,
            "y": Y,
        }


@dataclass
class FrontApertureConfig(_ConfigMixin):
    aperture_method: Literal["circular", "rectangular"] | None = "circular"
    aperture_thickness: float = 0.01  # m
    aperture_center: tuple[int, int] = (0, 0)  # px
    aperture_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.aperture_thickness <= 0:
            raise ValueError(
                f"aperture_thickness must be positive, got {self.aperture_thickness}"
            )

    def setup(
        self, shape: tuple[int, int], real_space_pixel_size: float
    ) -> structures.Apertures2D:
        aperture = structures.Apertures2D(shape, real_space_pixel_size)
        if self.aperture_method == "circular":
            aperture.create_circle_aperture(
                center=self.aperture_center,
                **self.aperture_config,
            )
        return aperture


@dataclass
class IlluminationConfig(_ConfigMixin):
    illumination_function: Literal["gaussian"] | None = "gaussian"
    illumination_center: tuple[int, int] = (0, 0)  # px
    illumination_config: dict = field(default_factory=dict)

    def setup(
        self,
        beam_params: light_beam.beam_parameters,
        shape: tuple[int, int],
        real_space_pixel_size: float,
    ) -> light_beam.illumination:
        illum = light_beam.illumination(beam_params, shape, real_space_pixel_size)

        if self.illumination_function == "gaussian":
            illum.gauss_beam(
                center=self.illumination_center, **self.illumination_config
            )
        elif self.illumination_function is None:
            illum.plane_wave(shape)
        return illum


@dataclass
class SampleConfig(_ConfigMixin):
    recipe: str = "Recipe"
    other_config: dict = field(default_factory=dict)

    def setup(self) -> structures.MultilayerRecipe:
        return structures.parse_recipe(self.recipe, **self.other_config)


@dataclass
class MagneticPatternConfig(_ConfigMixin):
    pattern_type_method: str = "skyrmion_pattern"
    pattern_config: dict = field(default_factory=dict)

    def setup(self, shape: tuple[int, int], real_space_pixel_size: float):
        _methods = {
            "skyrmion_pattern": pattern_generator.create_skyrmion_pattern,
        }
        method = _methods.get(self.pattern_type_method)
        if method is None:
            raise ValueError(
                f"Unknown pattern_type_method: {self.pattern_type_method!r}"
            )
        return method(
            sz_array=list(shape),
            real_space_pixel_size=real_space_pixel_size,
            **self.pattern_config,
        )
