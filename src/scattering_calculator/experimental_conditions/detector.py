from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
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
        """Compute sparse real-space (x, y) coordinate grids for the detector plane.

        Sets ``self.detx`` and ``self.dety`` as sparse 2-D arrays of physical
        coordinates in metres, centred on the optical axis. They broadcast to
        the full detector shape when q-space coordinates are calculated.
        """
        x = (
            np.arange(self.detector_shape[1]) - self.detector_center[1]
        ) * self.pixel_size
        y = (
            np.arange(self.detector_shape[0]) - self.detector_center[0]
        ) * self.pixel_size

        X, Y = np.meshgrid(x, y, sparse=True)
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
        self.beamstop = beamstop_mask

    def visualize_beamstop(self) -> None:
        """Visualize the beamstop mask in both pixel and real-space coordinates."""
        extent_det_real = self.get_detector_extent_real_space()

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(self.beamstop)
        ax[0].set_title("Beamstop in px")
        ax[1].imshow(self.beamstop, extent=1e3 * extent_det_real)
        ax[1].set_title("Beamstop in mm")
        ax[1].set_xlabel("x in mm")
        ax[1].set_ylabel("y in mm")

    def calc_resolution_from_detector(self) -> float:
        """Calculate the real-space resolution corresponding to the detector's maximum q.

        Returns
        -------
        resolution : float
            Real-space resolution in metres.
        """
        q_max = np.max(self.detqx)-np.min(self.detqx)
        #np.sqrt(np.max(self.detqx**2 + self.detqy**2))
        resolution = 2 * np.pi / q_max
        self.real_space_resolution = resolution
        return resolution


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

    def _project_length_to_detector_pixels(self, length: float | None) -> float | None:
        """Project a beamstop-plane length in metres to detector pixels."""
        if length is None:
            return None
        return self.calc_effective_beamstop_radius(length) / self.detector_pixel_size

    def create_circle_beamstop(
        self,
        center: tuple[float, float],
        radius: float | None = None,
        use_real_space_coordinates: bool = False,
        sigma: float | None = None,
        angle: float | None = None,
        ellipticity: tuple[float, float] = (1.0, 1.0),
        roughness: float = 0.0,
        roughness_modes: tuple[int, int] = (0, 0),
        wire_width: float = 0.0,
        wire_bend: float = 0.0,
        antialias: int = 1,
        seed: int | None = None,
        theta: float | None = None,
        ellipticity_range: tuple[float, float] | None = None,
    ) -> None:
        """Create a beamstop mask and store it in ``self.beamstop``.

        Parameters
        ----------
        center : tuple of float
            Mask centre coordinates (y, x) in pixels.
        radius : float or None
            Beamstop radius in metres. If ``None``, create an empty beamstop.
        use_real_space_coordinates : bool, optional
            If ``True``, convert the effective radius from metres to pixels
            using the detector pixel size. Default is ``False``.
        sigma : float or None, optional
            Standard deviation for Gaussian edge smoothing. When
            ``use_real_space_coordinates`` is ``True``, this is interpreted in
            metres at the beamstop plane, like ``radius`` and ``wire_width``.
            No smoothing when ``None``.
        angle : float or None, optional
            Beamstop and wire angle in radians. Random when ``None``.
        ellipticity : tuple of float, optional
            Random range for the ellipse axis ratio. ``(1, 1)`` creates a
            perfect circle.
        roughness : float, optional
            Relative amplitude of the smooth random boundary roughness.
        roughness_modes : tuple of int, optional
            Inclusive range of angular Fourier modes used for the boundary
            roughness.
        wire_width : float, optional
            Wire width. A value of ``0`` disables the wire.
        wire_bend : float, optional
            Maximum wire bend amplitude.
        antialias : int, optional
            Supersampling factor used while rasterising the beamstop and wire.
            Values greater than 1 average sub-pixels back to detector pixels,
            reducing stair-step artifacts for thin or diagonal features.
        seed : int or None, optional
            Random seed for reproducible beamstops.
        theta : float or None, optional
            Deprecated alias for ``angle``.
        ellipticity_range : tuple of float or None, optional
            Deprecated alias for ``ellipticity``.
        """
        if radius is None:
            self.create_empty_beamstop()
            return

        radius_effective = self.calc_effective_beamstop_radius(radius)
        wire_width_effective = self.calc_effective_beamstop_radius(wire_width)
        wire_bend_effective = self.calc_effective_beamstop_radius(wire_bend)
        sigma_effective = sigma

        if use_real_space_coordinates:
            radius_effective = radius_effective / self.detector_pixel_size
            wire_width_effective = wire_width_effective / self.detector_pixel_size
            wire_bend_effective = wire_bend_effective / self.detector_pixel_size
            sigma_effective = self._project_length_to_detector_pixels(sigma)

        rng = np.random.default_rng(seed)
        if angle is None:
            angle = theta
        if angle is None:
            angle = rng.uniform(0.0, np.pi)
        if ellipticity_range is not None:
            ellipticity = ellipticity_range
        orientation = angle
        antialias = max(1, int(antialias))

        if antialias > 1:
            rows, cols = self.detector_shape
            sample_y = (
                (np.arange(rows * antialias, dtype=float) + 0.5) / antialias
                - 0.5
            )
            sample_x = (
                (np.arange(cols * antialias, dtype=float) + 0.5) / antialias
                - 0.5
            )
            raster_shape = (rows * antialias, cols * antialias)
        else:
            sample_y = np.arange(self.detector_shape[0], dtype=float)
            sample_x = np.arange(self.detector_shape[1], dtype=float)
            raster_shape = self.detector_shape

        dy = sample_y[:, None] - center[0]
        dx = sample_x[None, :] - center[1]

        axis_ratio = rng.uniform(*ellipticity)
        radius_y = radius_effective * np.sqrt(axis_ratio)
        radius_x = radius_effective / np.sqrt(axis_ratio)

        cos_orientation = np.cos(orientation)
        sin_orientation = np.sin(orientation)
        xr = cos_orientation * dx + sin_orientation * dy
        yr = -sin_orientation * dx + cos_orientation * dy
        normalized_radius = np.sqrt((xr / radius_x) ** 2 + (yr / radius_y) ** 2)
        polar_angle = np.arctan2(yr / radius_y, xr / radius_x)

        boundary = np.ones(raster_shape, dtype=float)
        if roughness > 0:
            min_mode, max_mode = roughness_modes
            for mode in range(max(1, min_mode), max_mode + 1):
                amplitude = rng.normal(scale=roughness / mode)
                phase = rng.uniform(0.0, 2.0 * np.pi)
                boundary += amplitude * np.cos(mode * polar_angle + phase)
            boundary = np.clip(
                boundary, 1.0 - 2.0 * roughness, 1.0 + 2.0 * roughness
            )

        mask = (normalized_radius <= boundary).astype(float)

        if wire_width_effective > 0:
            along_wire = cos_orientation * dx + sin_orientation * dy
            across_wire = -sin_orientation * dx + cos_orientation * dy
            detector_diagonal = np.hypot(*self.detector_shape)
            bend_phase = rng.uniform(0.0, 2.0 * np.pi)
            bend_period = rng.uniform(0.8, 1.4) * detector_diagonal
            bend = wire_bend_effective * (
                np.sin(2.0 * np.pi * along_wire / bend_period + bend_phase)
                - np.sin(bend_phase)
            )
            wire_mask = np.abs(across_wire - bend) <= (0.5 * wire_width_effective)
            mask = np.maximum(mask, wire_mask.astype(float))

        if antialias > 1:
            rows, cols = self.detector_shape
            mask = mask.reshape(rows, antialias, cols, antialias).mean(axis=(1, 3))

        if sigma_effective is not None and sigma_effective != 0:
            mask = gaussian_filter(mask, sigma_effective)

        self.beamstop = mask

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
    DEFAULT_MEASUREMENT_CONFIG = {
        "exposure_time": 1.0,
        "number_frames": 1,
        "max_counts_per_image": None,
    }
    DEFAULT_DETECTOR_PARAMS = {
        "readout_noise_average": 50,
        "readout_noise_sigma": 3,
        "detector_threshold": 64e3,
        "counts_per_photon": 100,
        "quantum_efficiency": 1.0,
        "noise_seed": None,
    }
    DEFAULT_ARTIFACTS_CONFIG = {
        "counts_per_photon": 100,
        "sigma_photon": 0.75,
        "photon_n_classes": 1,
        "photon_n_variants": 30,
        "photon_kernel_size": 9,
        "photon_irregularity": 2.0,
        "regenerate_photon_kernels": True,
    }

    def __init__(
        self,
        detector_layout,
        hologram,
        beam_parameters,
        real_space_pixel_size,
        beamstop,
        artifacts_config=None,
        measurement_config=None,
        detector_params=None,
        coherence_length=None,
    ):
        self.detector_layout = detector_layout
        self.hologram = hologram
        self.beam_parameters = beam_parameters
        self.real_space_pixel_size = real_space_pixel_size
        self.beamstop = beamstop
        self.coherence_length = coherence_length

        self._apply_measurement_config(measurement_config)
        self._apply_artifacts_config(artifacts_config)
        self._apply_detector_params(detector_params)
        self.sigma_y = 0.0
        self.sigma_x = 0.0

    def _apply_config(self, defaults, config, allowed_keys, aliases=None):
        aliases = aliases or {}
        merged = dict(defaults)
        if config is not None:
            for key, value in config.items():
                attr = aliases.get(key, key)
                if attr not in allowed_keys:
                    raise ValueError(
                        f"Unknown detector_hologram parameter '{key}'. "
                        f"Allowed keys are: {sorted(allowed_keys | set(aliases))}"
                    )
                merged[attr] = value
        for key, value in merged.items():
            attr = aliases.get(key, key)
            setattr(self, attr, value)

    def _apply_measurement_config(self, measurement_config):
        self._apply_config(
            self.DEFAULT_MEASUREMENT_CONFIG,
            measurement_config,
            allowed_keys={
                "exposure_time",
                "number_frames",
                "max_counts_per_image",
            },
        )

    def _apply_artifacts_config(self, artifacts_config):
        self._apply_config(
            {
                **self.DEFAULT_ARTIFACTS_CONFIG,
                "photon_tile_size": self.hologram.shape[0],
            },
            artifacts_config,
            allowed_keys={
                "counts_per_photon",
                "sigma_photon",
                "photon_n_classes",
                "photon_n_variants",
                "photon_tile_size",
                "photon_kernel_size",
                "photon_irregularity",
                "regenerate_photon_kernels",
                "photon_sigma_range",
                "photon_ellipticity_range",
                "photon_class_seed",
                "photon_kernel_seed",
            },
        )

    def _apply_detector_params(self, detector_params):
        self._apply_config(
            self.DEFAULT_DETECTOR_PARAMS,
            detector_params,
            allowed_keys={
                "readout_noise_average",
                "readout_noise_sigma",
                "detector_threshold",
                "counts_per_photon",
                "quantum_efficiency",
                "noise_seed",
            },
            aliases={
                "noise_rms": "readout_noise_sigma",
            },
        )

    def _coherence_length_yx(self):
        coherence_length = self.coherence_length
        if coherence_length is None:
            coherence_length = getattr(self.beam_parameters, "coherence_length", None)
        if coherence_length is None:
            return None
        if np.isscalar(coherence_length):
            return float(coherence_length), float(coherence_length)
        if len(coherence_length) != 2:
            raise ValueError(
                "coherence_length must be a scalar or a 2-tuple "
                "(coherence_length_y, coherence_length_x)."
            )
        return float(coherence_length[0]), float(coherence_length[1])

    def _set_coherence_sigmas(self, hologram_shape):
        coherence_length = self._coherence_length_yx()
        if coherence_length is None:
            self.sigma_y = 0.0
            self.sigma_x = 0.0
            return

        coherence_length_y, coherence_length_x = coherence_length
        if coherence_length_y <= 0 or coherence_length_x <= 0:
            raise ValueError(
                "coherence_length values must be positive, got "
                f"{coherence_length}."
            )

        real_space_resolution = getattr(
            self.detector_layout, "real_space_resolution", None
        )
        if real_space_resolution is None:
            real_space_resolution = self.detector_layout.calc_resolution_from_detector()

        self.sigma_y = (
            hologram_shape[0] * real_space_resolution / coherence_length_y
        )
        self.sigma_x = (
            hologram_shape[0] * real_space_resolution / coherence_length_x
        )

    def add_noise(
        self,
        apply_beamstop_mask: bool = True,
        apply_detector_threshold: bool = True,
        store_no_beamstop: bool = False,
    ):
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
        rng = np.random.default_rng(getattr(self, "noise_seed", None))
        holo = np.array(self.hologram_detector, dtype=float, copy=True)
        if self.max_counts_per_image is None:
            holo *= self.exposure_time * self.quantum_efficiency

        npx, npy = holo.shape
        self._set_coherence_sigmas(holo.shape)

        # 1. convolution with a gaussian to simulate vibrations and partial transversal coherence
        if (self.sigma_x > 0) or (self.sigma_y > 0):
            kernel = np.outer(
                signal.windows.gaussian(npx, self.sigma_y),
                signal.windows.gaussian(npx, self.sigma_x),
            )
            if kernel.sum() > 0:
                kernel /= kernel.sum()
                holo = signal.fftconvolve(holo, kernel, mode="same")

        # 2. optionally rescale to a requested maximum detector count per image.
        # If None, preserve the physical scale set by photon flux and exposure time.
        if self.max_counts_per_image is not None:
            max_unblocked = np.amax((1.0 - self.beamstop.beamstop) * holo)
            if max_unblocked > 0:
                holo *= self.max_counts_per_image / max_unblocked

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
        detected_counts = photon_counts.astype(float) * self.counts_per_photon

        # 9. generate gaussian readout noise from detector once, then reuse it
        # for any requested output variants from this same detector realization.
        readout_noise = 0.0
        if self.readout_noise_average > 0 or self.readout_noise_sigma > 0:
            readout_noise = rng.normal(
                self.readout_noise_average * self.number_frames,
                self.readout_noise_sigma * np.sqrt(self.number_frames),
                detected_counts.shape,
            )

        def _finalize_variant(
            source_counts: np.ndarray,
            *,
            use_beamstop: bool,
            use_threshold: bool,
        ) -> np.ndarray:
            holo_variant = np.array(source_counts, dtype=float, copy=True)

            # 8. apply beamstop mask to shadow
            if use_beamstop:
                holo_variant *= 1.0 - self.beamstop.beamstop

            # 9. add gaussian readout noise from detector
            holo_variant += readout_noise

            # 10. round to integers and optionally cap image at thresholding camera value
            holo_variant = np.round(holo_variant, 0)
            if use_threshold:
                holo_variant = np.minimum(
                    holo_variant,
                    self.number_frames * self.detector_threshold,
                )

            # 11. divide by frame number: it is an average
            holo_variant /= self.number_frames

            # 12. just making sure the final product is positive
            holo_variant[holo_variant < 0] = 0
            return holo_variant

        self.hologram_exp = _finalize_variant(
            detected_counts,
            use_beamstop=apply_beamstop_mask,
            use_threshold=apply_detector_threshold,
        )
        if store_no_beamstop:
            self.hologram_exp_no_beamstop = _finalize_variant(
                detected_counts,
                use_beamstop=False,
                use_threshold=False,
            )

    def gnomonic_projection(self) -> NDArray[np.float64]:
        """Apply gnomonic projection to the hologram to correct for curvature of the Ewald sphere.
        Returns
        -------
            hologram_gnomonic : ndarray of shape (Ny, Nx)
            Gnomonic-projected hologram.
        """

        # Full reciprocal-space span of the shifted FFT grid. One FFT pixel is 1/self.real_space_pixel_size
        # therefore Dq / N, so q maps to q / Dq * N + N / 2.
        Dq = 2.0 * np.pi / self.real_space_pixel_size

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
