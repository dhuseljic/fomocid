from __future__ import annotations

import scipy.constants
from typing import Literal


def photon_energy_wavelength(
    value: float,
    unit: Literal["eV", "m"] = "eV",
) -> float:
    """Convert between photon energy and wavelength.

    Parameters
    ----------
    value : float
        Value to convert.
    Returns
    -------
    result : float
        Wavelength in metres or energy in eV
    """

    lambda_Xray = (
        scipy.constants.h * scipy.constants.c / (value * scipy.constants.e)
    )
    return lambda_Xray
