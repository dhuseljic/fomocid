from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scattering_calculator.utils.masking import circle_mask
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates


class detector_layout:
    """Detector geometry for a coherent scattering experiment.

    Parameters
    ----------
    pixel_size : float
        Physical pixel size in metres.
    detector_shape : tuple of int
        Detector dimensions (rows, cols) in pixels, e.g. ``(2048, 2048)``.
    distance_sample_detector : float
        Sample-to-detector distance in metres.
    detector_center : tuple of float
        Coordinates of the detector center in pixels, e.g. ``(1024, 1024)``.
    """

    def __init__(
        self,
        pixel_size: float = 10e-6,
        detector_shape: tuple[int, int] = (2048, 2048),
        distance_sample_detector: float = 0.15,
        detector_center: tuple[float, float] = (1024, 1024),
    ) -> None:
        self.pixel_size = pixel_size
        self.detector_shape = detector_shape
        self.distance_sample_detector = distance_sample_detector
        self.detector_center = detector_center

        # Calculate real-space coordinates of detector plane in meters
        self.calc_real_space_coordinates()

    def calc_real_space_coordinates(self) -> None:
        """Compute real-space (x, y) coordinate grids for the detector plane.

        Sets ``self.detx`` and ``self.dety`` as 2-D arrays of physical
        coordinates in metres, centred on the optical axis.
        """
        x = (
            np.arange(self.detector_shape[1]) - self.detector_center[1]
        ) * self.pixel_size
        y = (
            np.arange(self.detector_shape[0]) - self.detector_center[0]
        ) * self.pixel_size
        print(np.amin(x), np.amax(x), self.detector_center, self.pixel_size)
        X, Y = np.meshgrid(x, y)
        self.detx = X
        self.dety = Y

    def calc_q_space_coordinates(self, beam_parameters) -> None:
        """Compute Fourier-space (qx, qy) coordinate grids for the detector plane.

        Sets ``self.detqx`` and ``self.detqy`` as 2-D arrays of physical
        coordinates in metres, centred on the optical axis.
        """

        r = np.sqrt(self.detx**2 + self.dety**2)
        theta = np.arctan2(self.dety, self.detx)
        self.detqx = (
            beam_parameters.wavevector
            * np.sin(np.arctan(r / self.distance_sample_detector))
            * np.cos(theta)
        )
        self.detqy = (
            beam_parameters.wavevector
            * np.sin(np.arctan(r / self.distance_sample_detector))
            * np.sin(theta)
        )

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

    def visualize_beamstop(self) -> None:
        """Visualize the beamstop mask in both pixel and real-space coordinates."""
        extent_det_real = self.get_detector_extent_real_space()

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(self.beamstop_mask)
        ax[0].set_title("Beamstop in px")
        ax[1].imshow(self.beamstop_mask, extent=1e3 * extent_det_real)
        ax[1].set_title("Beamstop in mm")
        ax[1].set_xlabel("x in mm")
        ax[1].set_ylabel("y in mm")


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
        self.detector_shape = detector_config.detector_shape
        self.detector_pixel_size = detector_config.pixel_size
        self.distance_sample_detector = detector_config.distance_sample_detector
        self.distance_beamstop = distance_detector_beamstop
        self.beamstop = np.zeros(self.detector_shape)
        self.inverse_beamstop = np.ones(self.detector_shape)

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

        self.beamstop = circle_mask(
            self.detector_shape, center, radius_effective, sigma=sigma
        )

    def create_empty_beamstop(self) -> None:
        """Create an empty beamstop mask (all zeros)."""
        self.beamstop = np.zeros(self.detector_shape)

    def return_beamstop(self) -> NDArray[np.float64]:
        """Return the current beamstop mask array.

        Returns
        -------
        beamstop : ndarray
            2-D mask array of shape ``self.detector_shape``.
        """
        return self.beamstop


class detector_hologram:
    def __init__(
        self,
        detector_layout,
        hologram,
        beam_parameters,
        real_space_pixel_size,
        beamstop,
    ):
        self.detector_layout = detector_layout
        self.hologram = hologram
        self.beam_parameters = beam_parameters
        self.real_space_pixel_size = real_space_pixel_size
        self.beamstop = beamstop

        # acquisition details
        self.number_frames = 1
        self.max_counts_per_image = 60e3

        # detector readout
        self.readout_noise_average = 50
        self.readout_noise_sigma = 3
        self.detector_threshold = 64e3

        # photon-detector interaction details
        self.counts_per_photon = 100
        self.sigma_photon = 0.75
        self.photon_n_classes = 1
        self.photon_n_variants = 30
        self.photon_tile_size = self.hologram.shape[0]
        self.photon_kernel_size = 9
        self.photon_irregularity = 2.0
        self.regenerate_photon_kernels = True

        # beam properties
        self.sigma_y = 0.1
        self.sigma_x = 0.1

    def add_noise(self):
        """
        Given the hologram, the function simulates the holograms introducing drift,
         coherence effects and Poisson noise
        INPUT:
                readout_noise_average, readout_noise_sigma: readout noise of the camera
                sigma_h_px: sigma of drift in pixles. Takes into account also the spatial incoherence, so that has to be be taken ito accout too
                max_counts_per_image: max number of counts the camera can take in one image
                counts_per_photon:
                number_of_frames: number of acquired frames. The more, the lower the noise

        ----------
        Author: RB_2020
        """

        # 0. we start with holo, the FFT of the exit wave, hence the distribution of photons (or counts) at a certain point in the detector for a single image
        rng = np.random.default_rng()
        holo = self.hologram_detector
        npx, npy = holo.shape

        # 1. convolution with a gaussian to simulate vibrations and partial transversal coherence
        if (self.sigma_x > 0) or (self.sigma_y > 0):
            kernel = np.outer(
                signal.windows.gaussian(npx, self.sigma_y),
                signal.windows.gaussian(npx, self.sigma_x),
            )
            if kernel.sum() > 0:
                kernel /= kernel.sum()
                holo = signal.fftconvolve(holo, kernel, mode="same")

        # 2. adjust the maximum count to self.max_counts_per_image for a single frame.
        # holo is not the count number
        holo *= (self.max_counts_per_image) / np.amax(
            (1.0 - self.beamstop.beamstop) * holo
        )

        # 3. multiply by frame number to get counts of the entire set
        holo *= self.number_frames

        # 4. divide by counts_per_photons to get photon number
        expected_photons = holo / self.counts_per_photon
        expected_photons[expected_photons < 0] = 0

        # 5. Poisson photon sampling. Add poisson noise to number of photons
        photon_counts = rng.poisson(expected_photons).astype(np.int64)

        # 6. photon splatting with spatial kernel classes + event variants
        # this makes the photon events blobs affecting multiple pixels
        if self.sigma_photon > 0:
            # Create class map if it does not exist yet
            if not hasattr(self, "photon_class_map"):
                self.photon_class_map = self.make_tile_class_map(
                    holo.shape,
                    tile_size=getattr(self, "photon_tile_size", self.photon_tile_size),
                    n_classes=getattr(self, "photon_n_classes", self.photon_n_classes),
                    seed=getattr(self, "photon_class_seed", None),
                )

            # Create or regenerate kernel bank
            regenerate = getattr(self, "regenerate_photon_kernels", True)

            if regenerate or not hasattr(self, "photon_kernel_bank"):
                self.photon_kernel_bank = self.make_photon_kernel_bank(
                    n_classes=getattr(self, "photon_n_classes", self.photon_n_classes),
                    n_variants=getattr(
                        self, "photon_n_variants", self.photon_n_variants
                    ),
                    size=getattr(self, "photon_kernel_size", self.photon_kernel_size),
                    sigma_range=getattr(
                        self,
                        "photon_sigma_range",
                        (1.0 * self.sigma_photon, 1.5 * self.sigma_photon),
                    ),
                    ellipticity_range=getattr(
                        self,
                        "photon_ellipticity_range",
                        (0.6, 1.6),
                    ),
                    irregularity=getattr(
                        self, "photon_irregularity", self.photon_irregularity
                    ),
                    seed=getattr(self, "photon_kernel_seed", None),
                )

            photon_counts = self.photon_splat_with_spatial_classes(
                photon_counts=photon_counts,
                kernel_bank=self.photon_kernel_bank,
                class_map=self.photon_class_map,
                rng=rng,
            )

        # 7. convert photons back to detector counts
        holo = photon_counts * self.counts_per_photon

        # 8. apply beamstop mask to shadow
        holo *= 1.0 - self.beamstop.beamstop

        # 9. add gaussian readout noise from detector
        if self.readout_noise_average > 0 or self.readout_noise_sigma > 0:
            holo += np.random.normal(
                self.readout_noise_average * self.number_frames,
                self.readout_noise_sigma * np.sqrt(self.number_frames),
                holo.shape,
            )

        # 10. round to integers and cap image at thresholding camera value
        holo = np.round(holo, 0)
        holo = np.minimum(holo, self.number_frames * self.detector_threshold)

        # 11. divide by frame number: it is an average
        holo /= self.number_frames

        # 12. just making sure the final product is positive
        holo[holo < 0] = 0

        self.hologram_exp = holo.copy()

    def gnomonic_projection(self) -> NDArray[np.float64]:
        """Apply gnomonic projection to the hologram to correct for curvature of the Ewald sphere.
        Returns
        -------
            hologram_gnomonic : ndarray of shape (Ny, Nx)
            Gnomonic-projected hologram.
        """

        # how much is a pixel in q space
        Dq = np.pi / self.real_space_pixel_size

        ## we just need to rescale detqx so they are expressed in absolute pixel value
        self.hologram_detector = map_coordinates(
            self.hologram,
            [
                self.detector_layout.detqy / Dq * self.hologram.shape[0]
                + 1 * self.hologram.shape[0] / 2,
                self.detector_layout.detqx / Dq * self.hologram.shape[1]
                + 1 * self.hologram.shape[1] / 2,
            ],
            order=5,
            mode="constant",
            cval=0,
        )

    def make_tile_class_map(self, shape, tile_size=256, n_classes=32, seed=None):
        """
        Assign each detector tile to one of n_classes response classes.

        shape : (Ny, Nx)
        tile_size : int
        n_classes : int

        Returns
        -------
        class_map : (Ny, Nx) int
        """
        rng = np.random.default_rng(seed)

        Ny, Nx = shape
        nty = int(np.ceil(Ny / tile_size))
        ntx = int(np.ceil(Nx / tile_size))

        tile_classes = rng.integers(0, n_classes, size=(nty, ntx))

        class_map = np.zeros((Ny, Nx), dtype=np.int32)

        for iy in range(nty):
            for ix in range(ntx):
                y0 = iy * tile_size
                y1 = min((iy + 1) * tile_size, Ny)
                x0 = ix * tile_size
                x1 = min((ix + 1) * tile_size, Nx)
                class_map[y0:y1, x0:x1] = tile_classes[iy, ix]

        return class_map

    def make_random_photon_kernel(
        self,
        size=11,
        sigma_range=(0.7, 1.7),
        ellipticity_range=(0.6, 1.6),
        irregularity=0.20,
        seed=None,
    ):
        """
        Create one irregular single-photon response kernel.
        """
        rng = np.random.default_rng(seed)

        y, x = np.indices((size, size), dtype=float)
        cy = cx = (size - 1) / 2

        x -= cx
        y -= cy

        sigma_x = rng.uniform(*sigma_range)
        sigma_y = sigma_x * rng.uniform(*ellipticity_range)
        theta = rng.uniform(0, 2 * np.pi)

        xr = np.cos(theta) * x + np.sin(theta) * y
        yr = -np.sin(theta) * x + np.cos(theta) * y

        kernel = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))

        # Add smooth irregularity
        noise = 1 + irregularity * rng.normal(size=(size, size))
        noise = gaussian_filter(noise, sigma=1.0)
        kernel *= noise

        kernel[kernel < 0] = 0

        # Conserve photon number
        if kernel.sum() > 0:
            kernel /= kernel.sum()

        return kernel.astype(np.float64)

    def make_photon_kernel_bank(
        self,
        n_classes=32,
        n_variants=4,
        size=11,
        sigma_range=(0.7, 1.7),
        ellipticity_range=(0.6, 1.6),
        irregularity=0.20,
        seed=None,
    ):
        """
        Kernel bank with detector classes and event variants.

        Returns
        -------
        kernels : (n_classes, n_variants, size, size)
        """
        rng = np.random.default_rng(seed)

        kernels = np.empty((n_classes, n_variants, size, size), dtype=np.float64)

        for c in range(n_classes):
            for v in range(n_variants):
                kernels[c, v] = self.make_random_photon_kernel(
                    size=size,
                    sigma_range=sigma_range,
                    ellipticity_range=ellipticity_range,
                    irregularity=irregularity,
                    seed=rng.integers(0, 2**32 - 1),
                )

        return kernels

    def split_counts_into_variants(self, counts, n_variants, rng):
        """
        Split an integer photon-count image into n_variants images,
        preserving the total photon number per pixel.
        """
        counts = counts.astype(np.int64, copy=True)

        variants = []
        remaining = counts.copy()

        for v in range(n_variants - 1):
            p = 1.0 / (n_variants - v)
            take = rng.binomial(remaining, p)
            variants.append(take.astype(np.float64))
            remaining -= take

        variants.append(remaining.astype(np.float64))

        return variants

    def photon_splat_with_spatial_classes(
        self,
        photon_counts,
        kernel_bank,
        class_map,
        rng=None,
    ):
        """
        Apply spatially varying single-photon response using detector classes
        and random event variants.

        Parameters
        ----------
        photon_counts : (Ny, Nx)
            Integer photon counts per detector pixel.
        kernel_bank : (n_classes, n_variants, ky, kx)
            Photon response kernels.
        class_map : (Ny, Nx)
            Detector response class per pixel.
        rng : np.random.Generator or None

        Returns
        -------
        splatted : (Ny, Nx)
            Photon image after detector charge spreading.
        """
        if rng is None:
            rng = np.random.default_rng()

        photon_counts = np.asarray(photon_counts, dtype=np.int64)
        class_map = np.asarray(class_map, dtype=np.int32)
        kernel_bank = np.asarray(kernel_bank, dtype=np.float64)

        n_classes, n_variants = kernel_bank.shape[:2]

        splatted = np.zeros_like(photon_counts, dtype=np.float64)

        for c in range(n_classes):
            mask = class_map == c

            if not np.any(mask):
                continue

            class_counts = np.zeros_like(photon_counts, dtype=np.int64)
            class_counts[mask] = photon_counts[mask]

            if class_counts.sum() == 0:
                continue

            variant_counts = self.split_counts_into_variants(
                class_counts,
                n_variants=n_variants,
                rng=rng,
            )

            for v, counts_v in enumerate(variant_counts):
                if counts_v.sum() == 0:
                    continue

                kernel = kernel_bank[c, v]
                kernel = kernel / kernel.sum()

                splatted += signal.fftconvolve(counts_v, kernel, mode="same")

        return splatted
