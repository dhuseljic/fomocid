"""Simulation pipeline configuration for coherent scattering experiments."""

from .pipelines.hologram_pipeline import (
    HologramPipeline,
    HologramPipelineConfig,
    HologramPipelineRanges,
)
from .simulate_experiment import SetupSimulationExperiment
from .simulation_configuration import (
    BeamstopConfig,
    DetectorConfig,
    FrontApertureConfig,
    HologramConfig,
    IlluminationConfig,
    MagneticPatternConfig,
    SampleConfig,
    SamplePropagatorConfig,
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
    "HologramPipeline",
    "HologramPipelineConfig",
    "HologramPipelineRanges",
    "SetupSimulationExperiment",
    "BeamstopConfigRange",
    "Choice",
    "DetectorConfig",
    "DetectorConfigRange",
    "FrontApertureConfig",
    "FrontApertureConfigRange",
    "HologramConfig",
    "IlluminationConfig",
    "IlluminationConfigRange",
    "MagneticPatternConfig",
    "MagneticPatternConfigRange",
    "SampleConfig",
    "SampleConfigRange",
    "SamplePropagatorConfig",
    "SimulationConfig",
    "SimulationConfigRange",
    "Uniform",
    "XRayConfig",
    "XRayConfigRange",
]
