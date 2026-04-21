from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scattering_calculator.utils.masking import circle_mask


class detector_layout:
    """Detector geometry for a coherent scattering experiment.

    Parameters
    ----------
    pixel_size : float
        Physical pixel size in metres.
    shape : tuple of int
        Detector dimensions (rows, cols) in pixels, e.g. ``(2048, 2048)``.
    distance_sample_detector : float
        Sample-to-detector distance in metres.
    """

    def __init__(
        self,
        pixel_size: float = 10e-6,
        shape: tuple[int, int] = (2048, 2048),
        distance_sample_detector: float = 0.15,
    ) -> None:
        self.pixel_size = pixel_size
        self.shape = shape
        self.distance_sample_detector = distance_sample_detector

        # Calculate real-space coordinates of detector plane in meters
        self.calc_real_space_coordinates()

    def calc_real_space_coordinates(self) -> None:
        """Compute real-space (x, y) coordinate grids for the detector plane.

        Sets ``self.detx`` and ``self.dety`` as 2-D arrays of physical
        coordinates in metres, centred on the optical axis.
        """
        x = (np.arange(self.shape[1]) - self.shape[1] / 2) * self.pixel_size
        y = (np.arange(self.shape[0]) - self.shape[0] / 2) * self.pixel_size
        X, Y = np.meshgrid(x, y)
        self.detx = X
        self.dety = Y

    def get_detector_extent_real_space(self) -> NDArray[np.float64]:
        """Calculate the physical extent of the detector plane in metres.

        Returns
        -------
        extent : tuple of float
            Physical size of the detector plane in metres as (min_x, max_x, min_y, max_y).
        """
        extent_det_real = np.array(
            [
                np.min(self.detx),
                np.max(self.detx),
                np.min(self.dety),
                np.max(self.dety),
            ]
        )
        return extent_det_real

    def assign_beamstop(self, beamstop_mask: NDArray[np.float64]) -> None:
        """Attach a pre-computed beamstop mask to the detector.

        Parameters
        ----------
        beamstop_mask : ndarray
            Boolean or float mask with the same shape as the detector.
        """
        self.beamstop_mask = beamstop_mask


class beamstop:
    """Beamstop model for a coherent scattering experiment.

    The beamstop sits between the sample and detector. Its physical radius is
    projected to an effective radius on the detector plane accounting for the
    divergence geometry.

    Parameters
    ----------
    detector_config : detector_layout
        Detector configuration providing shape, pixel size, and
        sample-to-detector distance.
    distance_detector_beamstop : float
        Distance from the detector to the beamstop plane in metres.
    """

    def __init__(
        self,
        detector_config: detector_layout,
        distance_detector_beamstop: float,
    ) -> None:
        self.shape = detector_config.shape
        self.detector_pixel_size = detector_config.pixel_size
        self.distance_sample_detector = detector_config.distance_sample_detector
        self.distance_beamstop = distance_detector_beamstop
        self.beamstop = np.zeros(self.shape)
        self.inverse_beamstop = np.ones(self.shape)

    def calc_effective_beamstop_radius(self, radius: float) -> float:
        """Project a physical beamstop radius onto the detector plane.

        Accounts for the divergence of scattered beams between the beamstop
        and the detector.

        Parameters
        ----------
        radius : float
            Physical radius of the circular beamstop in metres.

        Returns
        -------
        effective_radius : float
            Projected radius on the detector plane in metres.
        """
        effective_radius = (
            radius
            * self.distance_sample_detector
            / (self.distance_sample_detector - self.distance_beamstop)
        )
        return effective_radius

    def create_circle_beamstop(
        self,
        center: tuple[float, float],
        radius: float,
        use_real_space_coordinates: bool = False,
        sigma: float | None = None,
    ) -> None:
        """Create a circular beamstop mask and store it in ``self.beamstop``.

        Parameters
        ----------
        center : tuple of int
            Mask centre coordinates (y, x) in pixels.
        radius : float
            Beamstop radius in metres.
        use_real_space_coordinates : bool, optional
            If ``True``, convert the effective radius from metres to pixels
            using the detector pixel size. Default is ``False``.
        sigma : float or None, optional
            Standard deviation for Gaussian edge smoothing. No smoothing when
            ``None``.
        """
        radius_effective = self.calc_effective_beamstop_radius(radius)

        if use_real_space_coordinates:
            radius_effective = radius_effective / self.detector_pixel_size

        self.beamstop = circle_mask(self.shape, center, radius_effective, sigma=sigma)

    def return_beamstop(self) -> NDArray[np.float64]:
        """Return the current beamstop mask array.

        Returns
        -------
        beamstop : ndarray
            2-D mask array of shape ``self.shape``.
        """
        return self.beamstop
