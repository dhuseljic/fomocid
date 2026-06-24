"""X-ray beam and illumination models for scattering simulations."""

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
        Polarization type: "CR" (circular right), "CL" (circular left),
        "LH" (linear horizontal), "LV" (linear vertical), or angle in radians
        for linear polarization at that angle. Legacy aliases "x" and "y" are
        accepted for backwards compatibility.

    Returns
    -------
    jones_vec : ndarray of shape (2,)
        Jones vector corresponding to the specified polarization.
    """

    if pol == "CR":
        return np.array([1, -1j]) / np.sqrt(2)
    elif pol == "CL":
        return np.array([1, 1j]) / np.sqrt(2)
    elif pol in ("LH", "x"):
        return np.array([1, 0])
    elif pol in ("LV", "y"):
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
        Polarization index (0 for Ex, 1 for Ey) or polarization type
        ("CR", "CL", "LH", "LV", or angle in radians). Legacy aliases
        "x" and "y" are accepted for backwards compatibility.

    Returns
    -------
    Jones wavefield : ndarray of shape (Ny, Nx, 2)
        Jones wavefield corresponding to the specified polarization.
    """
    return np.einsum("yx,s->yxs", scalar_wavefield, polarization_vector(pol))


def _normalize_alpha_beam(alpha_beam: float | ArrayLike) -> tuple[float, float]:
    """Return beam tilt as ``(alpha_y, alpha_x)`` in radians."""
    alpha = np.asarray(alpha_beam, dtype=float)
    if alpha.ndim == 0:
        return 0.0, float(alpha)
    if alpha.shape == (2,):
        return float(alpha[0]), float(alpha[1])
    raise ValueError(
        "alpha_beam must be a scalar or a two-value tuple "
        f"(alpha_y, alpha_x), got shape {alpha.shape}."
    )


def beam_direction_from_alpha(alpha_beam: float | ArrayLike) -> NDArray[np.float64]:
    """Return the propagation direction ``(x, y, z)`` for a beam tilt.

    ``alpha_beam`` follows the illumination convention ``(alpha_y, alpha_x)``
    in radians. A scalar remains the backward-compatible x-direction tilt.
    """
    alpha_y, alpha_x = _normalize_alpha_beam(alpha_beam)
    direction = np.array([np.tan(alpha_x), np.tan(alpha_y), 1.0], dtype=float)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"Invalid alpha_beam {alpha_beam!r}; beam direction is undefined.")
    return direction / norm


def gauss_beam(
    sz: tuple[int, int],
    px_size: float,
    center: ArrayLike,
    distance: float,
    fwhm: float,
    wavelength: float,
    alpha_beam: float | ArrayLike = (0.0, 0.0),
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
    alpha_beam : float or array-like of two floats, optional
        Beam tilt in radians. The preferred form is ``(alpha_y, alpha_x)``,
        where the two values tilt the beam along the sample y and x directions.
        A scalar is accepted for backward compatibility and is interpreted as
        ``(0, alpha_beam)``, matching the previous x-direction tilt behavior.
        ``0`` or ``(0, 0)`` evaluates the same beam cross-section as the
        historical normal-incidence implementation.

    Returns
    -------
    Gauss : complex ndarray of shape ``sz``
        Complex-valued transverse beam cross-section including phase.
    """

    ycenter = center[0]
    xcenter = center[1]

    alpha_y, alpha_x = _normalize_alpha_beam(alpha_beam)

    # Radial distance from beam centre in metres. Keep array shape as
    # (rows, cols) for rectangular fields.
    y, x = np.meshgrid(
        np.arange(sz[0], dtype=float),
        np.arange(sz[1], dtype=float),
        indexing="ij",
    )
    y = y - ycenter
    x = x - xcenter

    if alpha_y == 0 and alpha_x == 0:
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
        k = 2 * np.pi / wavelength

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

    y_idx, x_idx = np.meshgrid(
        np.arange(sz[0], dtype=float),
        np.arange(sz[1], dtype=float),
        indexing="ij",
    )
    x_m = (x_idx - xcenter) * px_size
    y_m = (y_idx - ycenter) * px_size
    x_beam = x_m * np.cos(alpha_x)
    y_beam = y_m * np.cos(alpha_y)
    z = distance + x_m * np.sin(alpha_x) + y_m * np.sin(alpha_y)
    rho = np.sqrt(x_beam**2 + y_beam**2)

    # Calc waist w(z) as function of distance z
    # position with respect to waist
    w0 = fwhm / (np.sqrt(2 * np.log(2)))
    # Waist radius
    zR = np.pi * w0**2 / wavelength
    # rayleigh range
    w = w0 * np.sqrt(1 + (z / zR) ** 2)

    # wavevector
    k = 2 * np.pi / wavelength

    # Transversal gaussian
    Gauss_trans = np.exp(-((rho / w) ** 2))

    # Longitudinal gaussian
    Gauss_long = np.ones_like(Gauss_trans, dtype=complex)
    nonzero_z = z != 0
    if np.any(nonzero_z):
        # Calc curvature R(z) as function of distance z
        R = np.empty_like(z, dtype=float)
        R[nonzero_z] = z[nonzero_z] * (1 + (zR / z[nonzero_z]) ** 2)

        # Calc Gouy phase
        Gouy = np.arctan(z / zR)

        Gauss_long[nonzero_z] = (w0 / w[nonzero_z]) * np.exp(
            -1j
            * (
                k * z[nonzero_z]
                + k * rho[nonzero_z] ** 2 / (2 * R[nonzero_z])
                - Gouy[nonzero_z]
            )
        )

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
        Polarization type, e.g. "CR", "CL", "LH", "LV", or angle in radians.
        Legacy aliases "x" and "y" are accepted for linear polarization.
    """

    def __init__(
        self,
        photon_energy: float,
        pol: str,
        photon_flux: float,
        coherence_length: tuple[float, float],
    ) -> None:
        """Initialize a beam_parameters instance.

        Parameters
        ----------
        photon_energy : float
            Input value for ``photon_energy``.
        pol : str
            Input value for ``pol``.
        photon_flux : float
            Input value for ``photon_flux``.
        coherence_length : tuple[float, float]
            Input value for ``coherence_length``.

        Returns
        -------
        None
            The function completes in place.
        """
        self.energy: float = photon_energy  # in eV
        self.wavelength: float = physics.photon_energy_wavelength(photon_energy)
        self.wavevector: float = (
            2 * np.pi / physics.photon_energy_wavelength(photon_energy)
        )
        self.pol = pol
        self.photon_flux: float = photon_flux  # in photons/s, set externally
        self.coherence_length: tuple[float, float] = coherence_length  # in m, (y, x)

    def calc_wavevector(self):
        """Run the calc wavevector operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        self.wavevector = 2 * np.pi / self.wavelength

    def return_params(self) -> beam_parameters:
        """Return the beam parameters object.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : beam_parameters
            Return value produced by the function.
        """
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
        """Initialize a illumination instance.

        Parameters
        ----------
        beam_parameters : beam_parameters
            Input value for ``beam_parameters``.
        sample_shape : tuple[int, int]
            Input value for ``sample_shape``.
        real_space_pixel_size : float
            Input value for ``real_space_pixel_size``.

        Returns
        -------
        None
            The function completes in place.
        """
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
        alpha_beam: float | ArrayLike = (0.0, 0.0),
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
        alpha_beam : float or array-like of two floats, optional
            Beam tilt in radians as ``(alpha_y, alpha_x)``. A scalar is accepted
            as the x-direction tilt for backward compatibility. ``0`` or
            ``(0, 0)`` keeps the historical normal-incidence Gaussian
            illumination exactly.

        Returns
        -------
        None
            The function completes in place.
        """
        self.illumination = gauss_beam(
            self.shape,
            self.pixel_size,
            np.asarray(center, dtype=float) / self.pixel_size
            + np.array([self.shape[0] / 2, self.shape[1] / 2]),
            distance,
            fwhm,
            self.beam_parameters.wavelength,
            alpha_beam=alpha_beam,
        )

    def get_illumination_jones(self) -> None:
        """Run the get illumination jones operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        self.illumination_jones = scalar_to_jones(
            self.illumination, self.beam_parameters.pol
        )

    def calc_real_space_coordinates(self) -> None:
        """Compute real-space (x, y) coordinate grids for the illumination plane.

        Sets ``self.x`` and ``self.y`` as 2-D arrays of physical
        coordinates in metres, centred on the optical axis.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
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

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
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
        """Set the beam cross-section to a plane wave with uniform amplitude and zero phase.

        Parameters
        ----------
        shape : tuple[int, int]
            Input value for ``shape``.

        Returns
        -------
        None
            The function completes in place.
        """
        self.illumination = np.ones(shape, dtype=complex)

    def return_illumination(self) -> NDArray[np.complex128] | None:
        """Return the current beam cross-section array.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : NDArray[np.complex128] | None
            Return value produced by the function.
        """
        return self.illumination

    def visualize_illumination(self) -> None:
        # Plot Gaussian beam
        """Run the visualize illumination operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
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
