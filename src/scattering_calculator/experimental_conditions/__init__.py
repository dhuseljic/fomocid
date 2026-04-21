"""Experimental setup configuration for coherent scattering simulations."""

from .detector import beamstop, detector_layout
from .light_beam import beam_parameters, gauss_beam, illumination

__all__ = [
    "beam_parameters",
    "beamstop",
    "detector_layout",
    "gauss_beam",
    "illumination",
]
