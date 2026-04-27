"""Simulation pipeline configuration for coherent scattering experiments."""

from .simulate_experiment import SimulationExperiment
from .simulation_configuration import (
    BeamstopConfig,
    DetectorConfig,
    FrontApertureConfig,
    IlluminationConfig,
    MagneticPatternConfig,
    SampleConfig,
    SimulationConfig,
    XRayConfig,
)
from .simulation_configuration_range import (
    BeamstopConfigRange,
    Choice,
    DetectorConfigRange,
    FrontApertureConfigRange,
    IlluminationConfigRange,
    MagneticPatternConfigRange,
    SampleConfigRange,
    SimulationConfigRange,
    Uniform,
    XRayConfigRange,
)

__all__ = [
    "BeamstopConfig",
    "SimulationExperiment",
    "BeamstopConfigRange",
    "Choice",
    "DetectorConfig",
    "DetectorConfigRange",
    "FrontApertureConfig",
    "FrontApertureConfigRange",
    "IlluminationConfig",
    "IlluminationConfigRange",
    "MagneticPatternConfig",
    "MagneticPatternConfigRange",
    "SampleConfig",
    "SampleConfigRange",
    "SimulationConfig",
    "SimulationConfigRange",
    "Uniform",
    "XRayConfig",
    "XRayConfigRange",
]
