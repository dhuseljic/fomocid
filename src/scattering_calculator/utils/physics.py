import scipy.constants


def photon_energy_wavelength(value, unit="eV"):
    """
    Conversion of different "photon energy" units

    Parameter
    =========
    value : scalar
        values that needs to be converted
    unit :
        Start unit, either 'eV' for energy in eV or 'nm' for wavelength in nm

    Returns
    =======
    lambda_Xray : scalar
        x-ray wavelength in nm
    energy_Xray : scalar
        x-ray energie in eV

    =======
    author: ck 2022
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
