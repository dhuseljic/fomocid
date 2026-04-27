import numpy as np

from scattering_calculator.experimental_conditions import detector, light_beam
from scattering_calculator.sample_generator import pattern_generator
from scattering_calculator.sample_generator import structures
from scattering_calculator.beam_propagator import Jones_propagator


class SetupSimulationExperiment:
    def __init__(
        xray_config,
        simulation_config,
        front_aperture_config,
        illumination_config,
        sample_config,
        magnetic_pattern_config,
        detector_config,
        beamstop_config,
    ):
        self.xray_config = xray_config
        self.simulation_config = simulation_config
        self.front_aperture_config = front_aperture_config
        self.illumination_config = illumination_config
        self.sample_config = sample_config
        self.magnetic_pattern_config = magnetic_pattern_config
        self.detector_config = detector_config
        self.beamstop_config = beamstop_config
