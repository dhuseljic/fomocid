from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scattering_calculator.utils import physics
import matplotlib.pyplot as plt


def polarization_vector(pol):
    """
    Return the Jones vector for a given polarization type.

    Parameters
    ----------
    pol : str or float
        Polarization type: "CR" (circular right), "CL" (circular left), "x" (linear horizontal), "y" (linear vertical), or angle in radians for linear polarization at that angle.

    Returns
    -------
    jones_vec : ndarray of shape (2,)
        Jones vector corresponding to the specified polarization.
    """

    if pol == "CR":
        return np.array([1, -1j]) / np.sqrt(2)
    elif pol == "CL":
        return np.array([1, 1j]) / np.sqrt(2)
    elif pol == "x":
        return np.array([1, 0])
    elif pol == "y":
        return np.array([0, 1])
    else:
        return np.array([np.sin(pol), np.cos(pol)])


def scalar_to_jones(scalar_wavefield, pol):
    """Calculate Jones wavefield for given polarization.

    Parameters
    ----------
    scalar_wavefield : ndarray of shape (Ny, Nx)
        Input scalar wavefield.
    pol : int or str
        Polarization index (0 for Ex, 1 for Ey) or polarization type ("CR", "CL", "x", "y", or angle in radians).

    Returns
    -------
    Jones wavefield : ndarray of shape (Ny, Nx, 2)
        Jones wavefield corresponding to the specified polarization.
    """
    return np.einsum("yx,s->yxs", scalar_wavefield, polarization_vector(pol))


def gauss_beam(
    sz: tuple[int, int],
    px_size: float,
    center: ArrayLike,
    distance: float,
    fwhm: float,
    wavelength: float,
) -> NDArray[np.complex128]:
    """Compute the cross-section of a Gaussian beam at a given propagation distance.

    Parameters
    ----------
    sz : tuple of int
        Output array shape (rows, cols), e.g. ``(512, 512)``.
    px_size : float
        Real-space pixel size in the sample plane in metres.
    center : array-like of float
        Beam centre coordinates ``[y_center, x_center]`` in pixels.
    distance : float
        Propagation distance from the beam waist in metres.
        Pass ``0`` to evaluate at the focus.
    fwhm : float
        Full-width at half-maximum of the beam at the waist in metres.
    wavelength : float
        Photon wavelength in metres.

    Returns
    -------
    Gauss : complex ndarray of shape ``sz``
        Complex-valued transverse beam cross-section including phase.
    """

    ycenter = center[0]
    xcenter = center[1]

    # Radial distance from beam centre in metres
    y, x = np.meshgrid(
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
    zR = np.pi * w0**2 / wavelength
    # rayleigh range
    w = w0 * np.sqrt(1 + (z / zR) ** 2)

    # wavevector
    k = 2 * np.pi / wlambda

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
    Gauss = Gauss / np.max(np.abs(Gauss))  # Normalize to max amplitude of 1

    return Gauss


class beam_parameters:
    """Incoming X-ray beam for a coherent scattering experiment.

    Stores photon energy, derived wavelength, and photon flux. Provides a
    convenience method to compute the Gaussian beam cross-section using the
    stored wavelength.

    Parameters
    ----------
    photon_energy : float
        Photon energy in eV. Used to compute ``self.wavelength``.
    pol: string or float
        Polarization "CR", "CL", "LH", "LV", float angle in radians
    Attributes
    ----------
    energy : float
        Photon energy in eV.
    wavelength : float
        Photon wavelength in metres derived from ``photon_energy``.
    illumination : complex ndarray
        Beam cross-section set by :meth:`gauss_beam`.
    pol : str
        Polarization type, e.g. "CR", "CL", "x", "y", or angle in radians.
    """

    def __init__(
        self,
        photon_energy: float,
        pol: str,
        photon_flux: float,
        coherence_length: float,
    ) -> None:
        self.energy: float = photon_energy  # in eV
        self.wavelength: float = physics.photon_energy_wavelength(photon_energy)
        self.wavevector: float = (
            2 * np.pi / physics.photon_energy_wavelength(photon_energy)
        )
        self.pol = pol
        self.photon_flux: float = photon_flux  # in photons/s, set externally
        self.coherence_length: float = coherence_length  # in m, set externally

    def calc_wavevector(self):
        self.wavevector = 2 * np.pi / self.wavelength

    def return_params(self) -> beam_parameters:
        """Return the beam parameters object."""
        return self.asdict(self)


class illumination:
    """Illumination field in the sample plane for a coherent scattering experiment.

    Computes and stores the complex-valued wavefield incident on the sample,
    given a set of beam parameters and the sample-plane geometry.

    Parameters
    ----------
    beam_parameters : beam_parameters
        Photon energy, wavelength, and flux of the incoming beam.
    sample_shape : tuple of int
        Shape of the sample plane array (rows, cols) in pixels.
    real_space_pixel_size : float
        Physical pixel size in the sample plane in metres.

    Attributes
    ----------
    beam_parameters : beam_parameters
        Reference to the beam parameter object.
    shape : tuple of int
        Sample plane dimensions in pixels.
    pixel_size : float
        Physical pixel size in metres.
    illumination : complex ndarray or None
        Complex wavefield; set by :meth:`gauss_beam` or :meth:`plane_wave`.
    x, y : ndarray
        2-D real-space coordinate grids in metres, set by
        :meth:`calc_real_space_coordinates`.
    """

    def __init__(
        self,
        beam_parameters: beam_parameters,
        sample_shape: tuple[int, int],
        real_space_pixel_size: float,
    ) -> None:
        self.beam_parameters = beam_parameters
        self.shape = sample_shape
        self.pixel_size = real_space_pixel_size
        self.pol = "CR"
        self.illumination: NDArray[np.complex128] | None = None
        self.illumination_jones: NDArray[np.complex128] | None = None
        # Calculate real-space coordinates of illumination plane in meters
        self.calc_real_space_coordinates()
        self.extent_real = self.get_illumination_extent_real_space()

    def gauss_beam(
        self,
        center: ArrayLike,
        distance: float,
        fwhm: float,
    ) -> None:
        """Compute the Gaussian beam cross-section and store in ``self.illumination``.

        Uses the wavelength stored in ``self.wavelength``.

        Parameters
        ----------
        center : array-like of float
            Beam centre coordinates ``[y_center, x_center]`` in meters, with respect to the sample
            (0,0) is the sample center
        distance : float
            Propagation distance from the beam waist in metres.
        fwhm : float
            Full-width at half-maximum of the beam at the waist in metres.
        """
        self.illumination = gauss_beam(
            self.shape,
            self.pixel_size,
            center / self.pixel_size + self.shape[0] // 2,
            distance,
            fwhm,
            self.beam_parameters.wavelength,
        )

    def get_illumination_jones(self) -> None:
        self.illumination_jones = scalar_to_jones(
            self.illumination, self.beam_parameters.pol
        )

    def calc_real_space_coordinates(self) -> None:
        """Compute real-space (x, y) coordinate grids for the illumination plane.

        Sets ``self.x`` and ``self.y`` as 2-D arrays of physical
        coordinates in metres, centred on the optical axis.
        """

        x = (np.arange(self.shape[1]) - self.shape[1] / 2) * self.pixel_size
        y = (np.arange(self.shape[0]) - self.shape[0] / 2) * self.pixel_size
        X, Y = np.meshgrid(x, y)
        self.x = X
        self.y = Y

    def get_illumination_extent_real_space(self) -> NDArray[np.float64]:
        """Calculate the physical extent of the illumination plane in metres.

        Returns
        -------
        extent : tuple of float
            Physical size of the detector plane in metres as (min_x, max_x, min_y, max_y).
        """

        self.extent_real = np.array(
            [
                np.min(self.x),
                np.max(self.x),
                np.min(self.y),
                np.max(self.y),
            ]
        )
        return self.extent_real

    def plane_wave(self, shape: tuple[int, int]) -> None:
        """Set the beam cross-section to a plane wave with uniform amplitude and zero phase."""
        self.illumination = np.ones(shape, dtype=complex)

    def return_illumination(self) -> NDArray[np.complex128] | None:
        """Return the current beam cross-section array."""
        return self.illumination

    def visualize_illumination(self) -> None:
        # Plot Gaussian beam
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        ma = np.max(np.abs(self.illumination) ** 2)
        ax[0].imshow(
            np.abs(self.illumination) ** 2,
            extent=1e6 * self.extent_real,
            vmin=0,
            vmax=ma,
        )
        ax[0].set_title("Intensity")
        ax[0].set_xlabel("x in µm")
        ax[0].set_ylabel("y in µm")
        ax[1].imshow(
            np.angle(self.illumination),
            extent=1e6 * self.extent_real,
            vmin=-np.pi,
            vmax=np.pi,
            cmap="hsv",
        )
        ax[1].set_title("Phase")
        ax[1].set_xlabel("x in µm")
        ax[1].set_ylabel("y in µm")
