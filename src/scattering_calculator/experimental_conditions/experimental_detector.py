import numpy as np
from scattering_calculator.utils.masking import circle_mask


class detector_layout:
    """
    Class to define the detector layout of scattering experiment

    Parameter
    =========
    pixel_size: scalar
        Pixel size in m
    shape: int tuple
        shape/dimension of detector in px, e.g., (2048,2048)

     =======
    """

    def __init__(
        self,
        pixel_size=10e-6,
        shape=(2048, 2048),
        distance_sample_detector=0.15,
        beamstop=None,
    ):
        self.pixel_size = pixel_size
        self.shape = shape
        self.distance_sample_detector = distance_sample_detector
        self.beamstop = beamstop


class beamstop:
    """
    Class to define the beamstop of scattering experiment

    Parameter
    =========
    radius: scalar
        radius of circular beamstop in m
     =======
    """

    def __init__(self, shape):
        self.shape = shape
        self.beamstop = np.zeros(shape)
        self.inverse_beamstop = np.ones(shape)

    def create_circle_beamstop(self, center, radius, sigma=None):
        self.beamstop = circle_mask(self.shape, center, radius, sigma=sigma)
        self.inverse_beamstop = 1 - self.beamstop

    def return_beamstop(self):
        return self.beamstop
