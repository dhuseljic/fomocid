import numpy as np
from scattering_calculator.utils import physics


def gauss_beam(sz, px_size, center, distance, fwhm, wlambda):
    """
    Cross-Section of gaussian beam in optics at 'distance' from focus

    Parameter
    =========
    sz: tuple of int
        Shape of new array, e.g., (2,3)
    px_size: scalar
        real-space pixel size in sample plane in m
    center: list
        Center coordiantes of beam [ycenter,xcenter] in px
    distance: scalar
        Distance between focus and plane of gaussian in m
    fwhm: scalar
        Full-width-half-maximum (FWHM) of gaussian in m
    wlambda: scalar
        photon wavelength in m

    Returns
    =======
    Gauss: complex array
        Cross-section Gaussian beam
    =======
    author: ck 2022
    """

    ycenter = center[0]
    xcenter = center[1]

    # Calc radius rho with respect to center
    rho = np.zeros(sz)
    # radius zylinder coordinates

    [y, x] = xs, ys = np.meshgrid(
        np.linspace(0, sz[0] - 1, sz[0]), np.linspace(0, sz[1] - 1, sz[1])
    )
    y = y - ycenter
    x = x - xcenter
    rho = np.sqrt(x**2 + y**2) * px_size

    # Calc waist w(z) as function of distance z
    z = distance
    # position with respect to waist
    w0 = fwhm / (np.sqrt(2 * np.log(2)))
    # Waist radius
    zR = np.pi * w0**2 / wlambda
    # rayleigh range
    w = w0 * np.sqrt(1 + (z / zR) ** 2)

    # wavevector
    k = 2 * np.pi / wlambda
    # wavevector

    # Transversal gaussian
    Gauss_trans = np.exp(-((rho / w) ** 2))

    # Longitudinal gaussian
    if z != 0:
        # Calc curvature R(z) as function of distance z
        R = z * (1 + (zR / z) ** 2)

        # Calc Gouy phase
        Gouy = np.arctan(z / zR)

        Gauss_long = w0 / w * np.exp(-1j * (k * z + k * rho**2 / (2 * R) - Gouy))
    elif z == 0:
        Gauss_long = 1

    Gauss = Gauss_trans * Gauss_long

    return Gauss


class wavefield:
    """
    Class to define the wavefield of the incoming light beam in scattering experiment

    Parameter
    =========
    photon_energy: scalar
        photon energy in eV, used to calculate wavelength
     =======
    """

    def __init__(self, photon_energy=50, photon_flux=1e12):
        self.energy = photon_energy  # in eV
        self.wavelength = physics.photon_energy_wavelength(photon_energy, unit="eV")
        self.photon_flux = photon_flux  # in photons/s

    def gauss_beam(self, sz, px_size, center, distance, fwhm):
        """Cross-Section of gaussian beam in optics at 'distance' from focus

        Parameter
        =========
        sz: tuple of int
            Shape of new array, e.g., (2,3)
        px_size: scalar
            real-space pixel size in sample plane in m
        center: list
            Center coordiantes of beam [ycenter,xcenter] in px
        distance: scalar
            Distance between focus and plane of gaussian in m
        fwhm: scalar
            Full-width-half-maximum (FWHM) of gaussian in m

        Returns
        =======
        Gauss: complex array
            Cross-section Gaussian beam
        =======
        """
        self.illumination = gauss_beam(
            sz, px_size, center, distance, fwhm, self.wavelength
        )
