from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scattering_calculator.simulation_pipelines.simulation_configuration import (
    BeamstopConfig,
    DetectorConfig,
    FrontApertureConfig,
    IlluminationConfig,
    MagneticPatternConfig,
    SampleConfig,
    SimulationConfig,
    XRayConfig,
    HologramConfig,
)


@dataclass(frozen=True)
class Uniform:
    """Continuous uniform distribution over [low, high].

    Parameters
    ----------
    low : float
        Lower bound (inclusive).
    high : float
        Upper bound (inclusive).
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(
                f"low must be less than high, got [{self.low}, {self.high}]"
            )

    def sample(self) -> float:
        """Draw one sample uniformly from [low, high]."""
        return float(np.random.uniform(self.low, self.high))


@dataclass(frozen=True)
class Choice:
    """Uniform random selection from a discrete set of options.

    Parameters
    ----------
    options : tuple
        Sequence of values to choose from.
    """

    options: tuple

    def __post_init__(self) -> None:
        if len(self.options) == 0:
            raise ValueError("options must not be empty")

    def sample(self) -> Any:
        """Draw one value uniformly from options."""
        return self.options[int(np.random.randint(len(self.options)))]


def _s(v: Any) -> Any:
    """Return v.sample() if v is a sampler, otherwise return v unchanged."""
    if isinstance(v, (Uniform, Choice)):
        return v.sample()
    return v


def _sample_dict(d: dict) -> dict:
    """Apply _s() to every value in *d*, recursively for nested dicts."""
    return {k: _sample_dict(v) if isinstance(v, dict) else _s(v) for k, v in d.items()}


@dataclass
class XRayConfigRange:
    """Parameter ranges for :class:`XRayConfig`.

    Each field accepts either a fixed value or a :class:`Uniform` / :class:`Choice`
    sampler. Call :meth:`sample` to draw one :class:`XRayConfig` instance.

    Examples
    --------
    >>> r = XRayConfigRange(energy=Uniform(770, 810), photon_flux=1e12)
    >>> cfg = r.sample()  # XRayConfig with energy drawn from U(770, 810)
    """

    energy: float | Uniform
    photon_flux: float | Uniform
    pol: str | Choice = "circular"
    coherence_length: float | Uniform = 10e-6  # m

    def sample(self) -> XRayConfig:
        """Sample one :class:`XRayConfig` from the defined ranges."""
        return XRayConfig(
            energy=_s(self.energy),
            photon_flux=_s(self.photon_flux),
            pol=_s(self.pol),
            coherence_length=_s(self.coherence_length),
        )


@dataclass
class SimulationConfigRange:
    """Parameter ranges for :class:`SimulationConfig`."""

    shape: tuple[int, int] | Choice = (256, 256)  # px
    real_space_pixel_size: float | Uniform = 10e-9  # m/px
    other_config: dict = field(default_factory=dict)

    def sample(self) -> SimulationConfig:
        """Sample one :class:`SimulationConfig` from the defined ranges."""
        return SimulationConfig(
            shape=_s(self.shape),
            real_space_pixel_size=_s(self.real_space_pixel_size),
            other_config=self.other_config,
        )


@dataclass
class FrontApertureConfigRange:
    """Parameter ranges for :class:`FrontApertureConfig`."""

    aperture_method: str | None | Choice = "circular"
    aperture_thickness: float | Uniform = 0.01  # m
    aperture_center: tuple[int, int] | Choice = (0, 0)  # px
    aperture_config: dict = field(default_factory=dict)

    def sample(self) -> FrontApertureConfig:
        """Sample one :class:`FrontApertureConfig` from the defined ranges."""
        return FrontApertureConfig(
            aperture_method=_s(self.aperture_method),
            aperture_thickness=_s(self.aperture_thickness),
            aperture_center=_s(self.aperture_center),
            aperture_config=self.aperture_config,
        )


@dataclass
class IlluminationConfigRange:
    """Parameter ranges for :class:`IlluminationConfig`."""

    illumination_function: str | None | Choice = "gaussian"
    illumination_center: tuple[int, int] | Choice = (0, 0)  # px
    illumination_config: dict = field(default_factory=dict)

    def sample(self) -> IlluminationConfig:
        """Sample one :class:`IlluminationConfig` from the defined ranges."""
        return IlluminationConfig(
            illumination_function=_s(self.illumination_function),
            illumination_center=_s(self.illumination_center),
            illumination_config=self.illumination_config,
        )


@dataclass
class SampleConfigRange:
    """Parameter ranges for :class:`SampleConfig`."""

    recipe: str | Choice = "Recipe"
    other_config: dict = field(default_factory=dict)

    def sample(self) -> SampleConfig:
        """Sample one :class:`SampleConfig` from the defined ranges."""
        return SampleConfig(
            recipe=_s(self.recipe),
            other_config=self.other_config,
        )


@dataclass
class MagneticPatternConfigRange:
    """Parameter ranges for :class:`MagneticPatternConfig`.

    Dict values inside ``pattern_config`` and ``pattern_config_length`` can
    be :class:`Uniform` or :class:`Choice` samplers, allowing individual
    pattern parameters to be swept independently.

    Examples
    --------
    >>> r = MagneticPatternConfigRange(
    ...     pattern_config={"angle_stripes": Uniform(0, np.pi)},
    ...     pattern_config_length={"stripe_width": Uniform(10e-9, 50e-9)},
    ... )
    >>> cfg = r.sample()  # MagneticPatternConfig with sampled stripe_width
    """

    pattern_type_method: str | Choice = "skyrmion_pattern"
    pattern_config: dict = field(default_factory=dict)
    pattern_config_length: dict = field(default_factory=dict)

    def sample(self) -> MagneticPatternConfig:
        """Sample one :class:`MagneticPatternConfig` from the defined ranges."""
        return MagneticPatternConfig(
            pattern_type_method=_s(self.pattern_type_method),
            pattern_config=_sample_dict(self.pattern_config),
            pattern_config_length=_sample_dict(self.pattern_config_length),
        )


@dataclass
class DetectorConfigRange:
    """Parameter ranges for :class:`DetectorConfig`.

    Examples
    --------
    >>> r = DetectorConfigRange(
    ...     sample_to_detector_distance=Uniform(0.05, 0.3),
    ...     detector_efficiency=Uniform(0.8, 1.0),
    ...     detector_noise_rms=Uniform(0.0, 5.0),
    ... )
    >>> cfg = r.sample()
    """

    shape: tuple[int, int] | Choice = (256, 256)  # px
    pixel_size: float | Uniform = 55e-6  # m/px
    sample_to_detector_distance: float | Uniform = 0.1  # m
    detector_center: tuple[int, int] | Choice = (0, 0)  # px
    detector_efficiency: float | Uniform = 1.0  # 0–1
    detector_noise_rms: float | Uniform = 0.0  # counts
    artifacts_method: str | None | Choice = None
    artifacts_config: dict = field(default_factory=dict)

    def sample(self) -> DetectorConfig:
        """Sample one :class:`DetectorConfig` from the defined ranges."""
        return DetectorConfig(
            shape=_s(self.shape),
            pixel_size=_s(self.pixel_size),
            sample_to_detector_distance=_s(self.sample_to_detector_distance),
            detector_center=_s(self.detector_center),
            detector_efficiency=_s(self.detector_efficiency),
            detector_noise_rms=_s(self.detector_noise_rms),
            artifacts_method=_s(self.artifacts_method),
            artifacts_config=self.artifacts_config,
        )


@dataclass
class BeamstopConfigRange:
    """Parameter ranges for :class:`BeamstopConfig`."""

    bs_method: str | None | Choice = "circular"
    bs_detector_distance: float | Uniform = 0.01  # m
    bs_center: tuple[int, int] | Choice = (0, 0)  # px
    bs_config: dict = field(default_factory=dict)

    def sample(self) -> BeamstopConfig:
        """Sample one :class:`BeamstopConfig` from the defined ranges."""
        return BeamstopConfig(
            bs_method=_s(self.bs_method),
            bs_detector_distance=_s(self.bs_detector_distance),
            bs_center=_s(self.bs_center),
            bs_config=self.bs_config,
        )
