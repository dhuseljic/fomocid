from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal


class _ConfigMixin:
    def to_dict(self) -> dict:
        """Return all configuration fields and their current values as a dict."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class XRayConfig(_ConfigMixin):
    energy: float  # eV
    photon_flux: float  # photons/s
    polarization: Literal["circular", "linear"] = "circular"
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class IlluminationConfig(_ConfigMixin):
    illumination_function: Literal["gaussian"] | None = "gaussian"
    illumination_center: tuple[int, int] = (0, 0)  # px
    illumination_config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SampleConfig(_ConfigMixin):
    recipe: str = "Recipe"
    other_config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MagneticPatternConfig(_ConfigMixin):
    pattern_type_method: str = "skyrmion_pattern"
    pattern_config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DetectorConfig(_ConfigMixin):
    shape: tuple[int, int] = (256, 256)  # px
    pixel_size: float = 55e-6  # m/px
    sample_to_detector_distance: float = 0.1  # m
    detector_center: tuple[int, int] = (0, 0)  # px
    detector_efficiency: float = 1.0  # 0–1
    detector_noise_rms: float = 0.0  # counts
    artifacts_method: str | None = None
    artifacts_config: dict = field(default_factory=dict)

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


@dataclass(frozen=True)
class BeamstopConfig(_ConfigMixin):
    bs_method: Literal["circular", "rectangular"] | None = "circular"
    bs_detector_distance: float = 0.01  # m
    bs_center: tuple[int, int] = (0, 0)  # px
    bs_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bs_detector_distance <= 0:
            raise ValueError(
                f"bs_detector_distance must be positive, got {self.bs_detector_distance}"
            )
