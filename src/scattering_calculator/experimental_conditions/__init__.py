"""Experimental setup configuration for coherent scattering simulations."""

from .detector import beamstop, detector_hologram, detector_layout
from .light_beam import (
    beam_parameters,
    gauss_beam,
    illumination,
    polarization_vector,
    scalar_to_jones,
)

__all__ = [
    "beam_parameters",
    "beamstop",
    "detector_hologram",
    "detector_layout",
    "gauss_beam",
    "illumination",
    "polarization_vector",
    "scalar_to_jones",
]
