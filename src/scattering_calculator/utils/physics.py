from __future__ import annotations

import scipy.constants
from typing import Literal


def photon_energy_wavelength(
    value: float,
    unit: Literal["eV", "nm"] = "eV",
) -> float:
    """Convert between photon energy and wavelength.

    Parameters
    ----------
    value : float
        Value to convert.
    unit : {"eV", "nm"}
        Unit of the input value. Pass ``"eV"`` to convert energy to wavelength
        in metres; pass ``"nm"`` to convert wavelength in nm to energy in eV.

    Returns
    -------
    result : float
        Wavelength in metres (when ``unit="eV"``) or energy in eV
        (when ``unit="nm"``).
    """

    if unit == "nm":
        lambda_Xray = (
            scipy.constants.h * scipy.constants.c / (value * scipy.constants.e)
        )
        return lambda_Xray
    elif unit == "eV":
        energy_Xray = (
            scipy.constants.h
            * scipy.constants.c
            / (value * 10 ** (-9) * scipy.constants.e)
        )
        return energy_Xray
