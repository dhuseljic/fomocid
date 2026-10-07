"""Simulation pipeline configuration for coherent scattering experiments."""

from .pipelines.hologram_pipeline import (
    HologramPipeline,
    HologramPipelineConfig,
    HologramPipelineRanges,
)
from .simulate_experiment import SetupSimulationExperiment
from .experiment import ExperimentConfig, OutputConfig, ScatteringExperiment
from .experiment_results import ExperimentResults, simulate_experiment, load_results, plot_results, run_experiment
from .simulation_configuration import (
    BeamstopConfig,
    DetectorConfig,
    FrontApertureConfig,
    HologramConfig,
    IlluminationConfig,
    SpectralComponentConfig,
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
    "ExperimentConfig",
    "OutputConfig",
    "ExperimentResults",
    "simulate_experiment",
    "load_results",
    "plot_results",
    "run_experiment",
    "ScatteringExperiment",
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
    "SpectralComponentConfig",
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
