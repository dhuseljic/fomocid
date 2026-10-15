"""Detector geometry, beamstops, noise, and sensor-artifact simulation."""

from __future__ import annotations
import numpy as np
from copy import copy
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

    def calc_q_space_coordinates(
        self,
        beam_parameters,
        ignore_flat_detector_curvature: bool = False,
    ) -> None:
        """Compute Fourier-space (qx, qy) coordinate grids for the detector plane.

        Sets ``self.detqx`` and ``self.detqy`` as 2-D arrays of physical
        coordinates in metres, centred on the optical axis.
        """

        if ignore_flat_detector_curvature:
            detqx = (
                beam_parameters.wavevector
                * self.detx
                / self.distance_sample_detector
            )
            detqy = (
                beam_parameters.wavevector
                * self.dety
                / self.distance_sample_detector
            )
            self.detqx, self.detqy = np.broadcast_arrays(detqx, detqy)
        else:
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
        antialias_method: str = "analytic",
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
        antialias_method : {"analytic", "supersample"}, optional
            ``"analytic"`` uses detector-resolution signed-distance coverage
            and avoids building a full supersampled detector grid. It is much
            faster for large detectors. ``"supersample"`` keeps the historical
            full-grid subpixel averaging. Default is ``"analytic"``.
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
        antialias_method = str(antialias_method)

        if antialias > 1 and antialias_method == "supersample":
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

        if antialias > 1 and antialias_method == "analytic":
            edge_scale = max(1e-12, min(abs(radius_x), abs(radius_y)))
            edge_distance = (boundary - normalized_radius) * edge_scale
            mask = np.clip(edge_distance + 0.5, 0.0, 1.0)
        else:
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
            wire_distance = 0.5 * wire_width_effective - np.abs(across_wire - bend)
            if antialias > 1 and antialias_method == "analytic":
                wire_mask = np.clip(wire_distance + 0.5, 0.0, 1.0)
            else:
                wire_mask = wire_distance >= 0
            mask = np.maximum(mask, wire_mask.astype(float))

        if antialias > 1 and antialias_method == "supersample":
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
        "nonlinear_response": None,
        "quantum_efficiency": 1.0,
        "noise_seed": None,
    }
    DEFAULT_ARTIFACTS_CONFIG = {
        "sigma_photon": 0.75,
        "photon_n_classes": 1,
        "photon_n_variants": 30,
        "photon_kernel_size": 9,
        "photon_irregularity": 2.0,
        "regenerate_photon_kernels": True,
        "camera_seed": None,
        "average_hot_pixels": 0.0,
        "average_cold_pixels": 0.0,
        "flicker_fraction": 0.0,
        "flicker_probability": 0.5,
        "hot_pixel_value": None,
        "hot_pixel_value_spread": 0.05,
        "hot_pixel_temporal_sigma": 0.02,
        "cold_pixel_value": 0.0,
        "cold_pixel_value_spread": 0.05,
        "cold_pixel_temporal_sigma": 0.02,
        "cosmic_rays_per_second": 0.0,
        "cosmic_ray_value": None,
        "cosmic_ray_value_spread": 0.15,
        "cosmic_ray_length_range": (2.0, 5.0),
        "cosmic_ray_aspect_ratio_range": (2.0, 3.0),
        "cosmic_ray_max_length": 5,
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
        self.photon_energy_eV = getattr(beam_parameters, 'energy', None)
        self.real_space_pixel_size = real_space_pixel_size
        self.beamstop = beamstop
        self.coherence_length = coherence_length

        artifacts_config, detector_params = self._normalize_counts_per_photon(
            artifacts_config,
            detector_params,
        )
        detector_params = self._normalize_detector_param_aliases(detector_params)
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

    @staticmethod
    def _normalize_counts_per_photon(artifacts_config, detector_params):
        """Keep ``counts_per_photon`` in detector parameters only.

        Older configurations sometimes placed this value in ``artifacts_config``.
        Accept that location as an input alias, but reject conflicting values
        instead of silently overwriting one with the other.
        """
        artifacts_config = dict(artifacts_config or {})
        detector_params = dict(detector_params or {})
        legacy_value = artifacts_config.pop("counts_per_photon", None)
        detector_value = detector_params.get("counts_per_photon")
        if legacy_value is not None:
            if detector_value is not None and detector_value != legacy_value:
                raise ValueError(
                    "Conflicting counts_per_photon values: use only "
                    "detector_params['counts_per_photon']."
                )
            detector_params["counts_per_photon"] = legacy_value
        return artifacts_config, detector_params

    @staticmethod
    def _normalize_detector_param_aliases(detector_params):
        """Normalize legacy detector parameter names without silent overwrite."""
        detector_params = dict(detector_params or {})
        legacy_noise = detector_params.pop("noise_rms", None)
        canonical_noise = detector_params.get("readout_noise_sigma")
        if legacy_noise is not None:
            if canonical_noise is not None and canonical_noise != legacy_noise:
                raise ValueError(
                    "Conflicting detector noise values: use only "
                    "detector_params['readout_noise_sigma']."
                )
            detector_params["readout_noise_sigma"] = legacy_noise
        return detector_params

    def _apply_artifacts_config(self, artifacts_config):
        artifacts_config = dict(artifacts_config or {})
        if (
            "cosmic_ray_max_length" in artifacts_config
            and "cosmic_ray_length_range" not in artifacts_config
        ):
            maximum = float(artifacts_config["cosmic_ray_max_length"])
            artifacts_config["cosmic_ray_length_range"] = (min(2.0, maximum), maximum)
        self._apply_config(
            {
                **self.DEFAULT_ARTIFACTS_CONFIG,
                "photon_tile_size": self.hologram.shape[0],
            },
            artifacts_config,
            allowed_keys={
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
                "camera_seed",
                "average_hot_pixels",
                "average_cold_pixels",
                "flicker_fraction",
                "flicker_probability",
                "hot_pixel_value",
                "hot_pixel_value_spread",
                "hot_pixel_temporal_sigma",
                "cold_pixel_value",
                "cold_pixel_value_spread",
                "cold_pixel_temporal_sigma",
                "cosmic_rays_per_second",
                "cosmic_ray_value",
                "cosmic_ray_value_spread",
                "cosmic_ray_length_range",
                "cosmic_ray_aspect_ratio_range",
                "cosmic_ray_max_length",
            },
        )

        for name in ("average_hot_pixels", "average_cold_pixels", "cosmic_rays_per_second"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not 0.0 <= self.flicker_fraction <= 1.0:
            raise ValueError("flicker_fraction must be between 0 and 1.")
        if not 0.0 <= self.flicker_probability <= 1.0:
            raise ValueError("flicker_probability must be between 0 and 1.")
        if self.cosmic_ray_max_length < 1:
            raise ValueError("cosmic_ray_max_length must be at least 1 pixel.")
        for name in (
            "hot_pixel_value_spread",
            "hot_pixel_temporal_sigma",
            "cold_pixel_value_spread",
            "cold_pixel_temporal_sigma",
            "cosmic_ray_value_spread",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        for name in ("cosmic_ray_length_range", "cosmic_ray_aspect_ratio_range"):
            low, high = getattr(self, name)
            if low <= 0 or high < low:
                raise ValueError(f"{name} must be a positive increasing pair.")
        if self.cosmic_ray_aspect_ratio_range[0] < 1:
            raise ValueError("cosmic_ray_aspect_ratio_range cannot start below 1.")

    def _camera_defect_map(self, shape):
        """Return the reproducible hot/cold pixel map for this camera seed.

        Parameters
        ----------
        shape : tuple of int
            Sensor image shape as ``(rows, columns)``.

        Returns
        -------
        dict
            Pixel coordinate arrays and sampled defect counts.
        """
        rng = np.random.default_rng(getattr(self, "camera_seed", None))
        pixel_count = int(np.prod(shape))
        n_hot = min(rng.poisson(getattr(self, "average_hot_pixels", 0.0)), pixel_count)
        n_cold = min(rng.poisson(getattr(self, "average_cold_pixels", 0.0)), pixel_count - n_hot)
        selected = rng.choice(pixel_count, size=n_hot + n_cold, replace=False)
        hot_flat = selected[:n_hot]
        cold_flat = selected[n_hot:]
        n_flicker = int(rng.binomial(n_hot, getattr(self, "flicker_fraction", 0.0)))
        flicker_flat = (
            rng.choice(hot_flat, size=n_flicker, replace=False)
            if n_flicker else np.empty(0, dtype=int)
        )
        fixed_hot_flat = np.setdiff1d(hot_flat, flicker_flat, assume_unique=False)
        threshold = float(getattr(self, "detector_threshold", 64e3))
        hot_center = getattr(self, "hot_pixel_value", None)
        hot_center = 0.9 * threshold if hot_center is None else float(hot_center)
        cold_center = float(getattr(self, "cold_pixel_value", 0.0))
        hot_baselines = np.clip(
            rng.normal(
                hot_center,
                abs(hot_center) * getattr(self, "hot_pixel_value_spread", 0.05),
                size=n_hot,
            ),
            0,
            None,
        )
        cold_baselines = np.clip(
            rng.normal(
                cold_center,
                max(abs(cold_center), 1.0)
                * getattr(self, "cold_pixel_value_spread", 0.05),
                size=n_cold,
            ),
            0,
            None,
        )
        hot_baseline_by_flat = dict(zip(hot_flat.tolist(), hot_baselines.tolist()))
        return {
            "hot": np.unravel_index(fixed_hot_flat, shape),
            "flicker": np.unravel_index(flicker_flat, shape),
            "cold": np.unravel_index(cold_flat, shape),
            "hot_baseline_values": np.asarray(
                [hot_baseline_by_flat[index] for index in fixed_hot_flat], dtype=float
            ),
            "flicker_baseline_values": np.asarray(
                [hot_baseline_by_flat[index] for index in flicker_flat], dtype=float
            ),
            "cold_baseline_values": cold_baselines,
            "n_hot": n_hot,
            "n_cold": n_cold,
        }

    def _add_sensor_artifacts(self, image, rng):
        """Apply persistent bad pixels and transient cosmic-ray tracks.

        Parameters
        ----------
        image : ndarray
            Detector-count image before sensor defects.
        rng : numpy.random.Generator
            Per-exposure random generator.

        Returns
        -------
        ndarray
            Copy of the image with sensor artifacts applied.
        """
        result = np.array(image, dtype=float, copy=True)
        defect_map = self._camera_defect_map(result.shape)
        self.camera_defect_map = defect_map
        hot_setting = getattr(self, "hot_pixel_value", None)
        hot_value = 0.9 * self.detector_threshold if hot_setting is None else hot_setting
        cosmic_setting = getattr(self, "cosmic_ray_value", None)
        cosmic_value = hot_value if cosmic_setting is None else cosmic_setting

        hot_baselines = defect_map["hot_baseline_values"]
        hot_sigma = np.abs(hot_baselines) * getattr(self, "hot_pixel_temporal_sigma", 0.02)
        result[defect_map["hot"]] = np.clip(
            rng.normal(
                hot_baselines * self.number_frames,
                hot_sigma * np.sqrt(self.number_frames),
            ),
            0,
            None,
        )
        cold_baselines = defect_map["cold_baseline_values"]
        cold_sigma = np.maximum(np.abs(cold_baselines), 1.0) * getattr(
            self, "cold_pixel_temporal_sigma", 0.02
        )
        result[defect_map["cold"]] = np.clip(
            rng.normal(
                cold_baselines * self.number_frames,
                cold_sigma * np.sqrt(self.number_frames),
            ),
            0,
            None,
        )
        flicker_y, flicker_x = defect_map["flicker"]
        if flicker_y.size:
            active_frames = rng.binomial(
                self.number_frames, getattr(self, "flicker_probability", 0.5), size=flicker_y.size
            )
            flicker_baselines = defect_map["flicker_baseline_values"]
            flicker_sigma = np.abs(flicker_baselines) * getattr(
                self, "hot_pixel_temporal_sigma", 0.02
            )
            result[flicker_y, flicker_x] = np.clip(
                rng.normal(
                    flicker_baselines * active_frames,
                    flicker_sigma * np.sqrt(active_frames),
                ),
                0,
                None,
            )

        exposure = 1.0 if self.exposure_time is None else float(self.exposure_time)
        if exposure < 0:
            raise ValueError("exposure_time must be non-negative or None.")
        expected_rays = getattr(self, "cosmic_rays_per_second", 0.0) * exposure * self.number_frames
        n_rays = int(rng.poisson(expected_rays))
        self.cosmic_ray_count = n_rays
        cosmic_ray_tracks = []
        if n_rays:
            rows, cols = result.shape
            for _ in range(n_rays):
                y0 = rng.uniform(0, rows - 1)
                x0 = rng.uniform(0, cols - 1)
                length_range = getattr(self, "cosmic_ray_length_range", None)
                if length_range is None:
                    length_range = (1.0, getattr(self, "cosmic_ray_max_length", 5))
                length = rng.uniform(*length_range)
                aspect_ratio = rng.uniform(
                    *getattr(self, "cosmic_ray_aspect_ratio_range", (2.0, 3.0))
                )
                angle = rng.uniform(0, 2 * np.pi)
                event_value = max(
                    0.0,
                    rng.normal(
                        cosmic_value,
                        abs(cosmic_value)
                        * getattr(self, "cosmic_ray_value_spread", 0.15),
                    ),
                )
                sigma_major = length / 2.355
                sigma_minor = sigma_major / aspect_ratio
                radius = max(2, int(np.ceil(3 * sigma_major)))
                y_min, y_max = max(0, int(y0) - radius), min(rows, int(y0) + radius + 1)
                x_min, x_max = max(0, int(x0) - radius), min(cols, int(x0) + radius + 1)
                yy, xx = np.indices((y_max - y_min, x_max - x_min), dtype=float)
                dy, dx = yy + y_min - y0, xx + x_min - x0
                major = np.sin(angle) * dy + np.cos(angle) * dx
                minor = np.cos(angle) * dy - np.sin(angle) * dx
                event = event_value * np.exp(
                    -0.5 * ((major / sigma_major) ** 2 + (minor / sigma_minor) ** 2)
                )
                result[y_min:y_max, x_min:x_max] += event
                cosmic_ray_tracks.append(
                    (y0, x0, length, aspect_ratio, angle, event_value)
                )
        self.cosmic_ray_tracks = np.asarray(cosmic_ray_tracks, dtype=float).reshape(-1, 6)
        return result

    def _apply_detector_params(self, detector_params):
        self._apply_config(
            self.DEFAULT_DETECTOR_PARAMS,
            detector_params,
            allowed_keys={
                "readout_noise_average",
                "readout_noise_sigma",
                "detector_threshold",
                "counts_per_photon",
                "nonlinear_response",
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

    def _gain_at_energy(self, energy):
        """Evaluate constant or energy-dependent ADU per detected photon.

        Parameters
        ----------
        energy : float or None
            Physical photon energy in eV; required only for a gain callable.

        Returns
        -------
        float
            Finite positive gain, evaluated independently for each color.
        """
        gain = self.counts_per_photon
        if callable(gain):
            if energy is None or not np.isfinite(energy) or energy <= 0:
                raise ValueError(
                    "Energy-dependent counts_per_photon requires a physical photon energy in eV"
                )
            gain = gain(float(energy))
        if np.iscomplexobj(gain):
            raise ValueError("counts_per_photon must be real ADU per photon")
        value = np.asarray(gain, dtype=float)
        if value.ndim != 0 or not np.isfinite(value) or value <= 0:
            raise ValueError("counts_per_photon must evaluate to a finite positive scalar")
        return float(value)

    def _expected_photon_components(self):
        """Prepare per-frame detected photon means and energy-specific gains.

        Parameters
        ----------
        None
            Uses ideal photon rates, exposure, efficiency and optional count budget.

        Returns
        -------
        list of tuple
            (gain_ADU_per_photon, expected_detected_photons) for each color.
        """
        energy = getattr(
            self,
            "photon_energy_eV",
            getattr(getattr(self, "beam_parameters", None), "energy", None),
        )
        components = getattr(
            self, "photon_rate_components", ((energy, self.hologram_detector),)
        )
        if not components:
            raise ValueError("At least one photon-rate component is required")
        exposure = 1.0 if self.exposure_time is None else float(self.exposure_time)
        if not np.isfinite(exposure) or exposure < 0:
            raise ValueError("exposure_time must be finite and non-negative or None")
        efficiency = np.asarray(self.quantum_efficiency, dtype=float)
        if not np.isfinite(efficiency).all() or np.any((efficiency < 0) | (efficiency > 1)):
            raise ValueError("quantum_efficiency must lie between zero and one")
        self._set_coherence_sigmas(self.hologram_detector.shape)
        prepared = []
        for energy, rate in components:
            gain = self._gain_at_energy(energy)
            rate = np.array(rate, dtype=float, copy=True)
            if (
                rate.shape != self.hologram_detector.shape
                or not np.isfinite(rate).all()
                or np.any(rate < 0)
            ):
                raise ValueError(
                    "Photon-rate components must match the ideal image and be finite/nonnegative"
                )
            if self.max_counts_per_image is None:
                rate *= exposure * efficiency
            if self.sigma_x > 0 or self.sigma_y > 0:
                rows, columns = rate.shape
                kernel = np.outer(
                    signal.windows.gaussian(rows, self.sigma_y),
                    signal.windows.gaussian(columns, self.sigma_x),
                )
                if kernel.sum() > 0:
                    rate = signal.fftconvolve(rate, kernel / kernel.sum(), mode="same")
            prepared.append((gain, np.maximum(rate, 0.0)))
        if self.max_counts_per_image is not None:
            budget = float(self.max_counts_per_image)
            if not np.isfinite(budget) or budget < 0:
                raise ValueError(
                    "max_counts_per_image must be finite and non-negative or None"
                )
            expected_adu = sum(gain * mean for gain, mean in prepared)
            shadow = 0.0 if self.beamstop is None else self.beamstop.beamstop
            peak = np.max((1.0 - shadow) * expected_adu)
            scale = budget / peak if peak > 0 else 0.0
            prepared = [(gain, mean * scale) for gain, mean in prepared]
        return prepared

    def _spread_photons(self, photon_counts, rng):
        """Distribute independently sampled photon events across sensor pixels.

        Parameters
        ----------
        photon_counts : ndarray
            Realized photons for one spectral component.
        rng : numpy.random.Generator
            Acquisition generator shared across the component sampling steps.

        Returns
        -------
        ndarray
            Photon-event response, before conversion to ADU.
        """
        if self.sigma_photon > 0:
            # Create class map if it does not exist yet
            if not hasattr(self, "photon_class_map"):
                self.photon_class_map = self.make_tile_class_map(
                    photon_counts.shape,
                    tile_size=getattr(self, "photon_tile_size", self.photon_tile_size),
                    n_classes=getattr(self, "photon_n_classes", self.photon_n_classes),
                    seed=getattr(self, "photon_class_seed", None),
                )

            # Create or regenerate kernel bank
            regenerate = getattr(self, "regenerate_photon_kernels", True)

            if regenerate or not hasattr(self, "photon_kernel_bank"):
                self.photon_kernel_bank = self.make_photon_kernel_bank(
                    n_classes=getattr(self, "photon_n_classes", self.photon_n_classes),
                    n_variants=getattr(self, "photon_n_variants", self.photon_n_variants),
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

        return photon_counts

    def _acquire_frames(
        self, rng, apply_beamstop_mask, apply_detector_threshold, store_no_beamstop
    ):
        """Convert, read out and clip each physical frame before averaging.

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator supplying independent per-frame acquisition seeds.
        apply_beamstop_mask, apply_detector_threshold, store_no_beamstop : bool
            Same acquisition controls as add_noise.

        Returns
        -------
        None
            Stores frame-averaged measurements and optional unmasked variant.
        """
        frame = copy(self)
        frame.number_frames = 1
        total = np.zeros_like(self.hologram_detector, dtype=float)
        unmasked = np.zeros_like(total) if store_no_beamstop else None
        tracks = []
        for _ in range(self.number_frames):
            frame.noise_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            frame.add_noise(
                apply_beamstop_mask, apply_detector_threshold, store_no_beamstop
            )
            total += frame.hologram_exp
            tracks.append(frame.cosmic_ray_tracks.copy())
            if store_no_beamstop:
                unmasked += frame.hologram_exp_no_beamstop
        self.hologram_exp = total / self.number_frames
        self.camera_defect_map = frame.camera_defect_map
        self.cosmic_ray_tracks = np.concatenate(tracks, axis=0)
        self.cosmic_ray_count = len(self.cosmic_ray_tracks)
        for name in ("photon_class_map", "photon_kernel_bank"):
            if hasattr(frame, name):
                setattr(self, name, getattr(frame, name))
        if store_no_beamstop:
            self.hologram_exp_no_beamstop = unmasked / self.number_frames

    def add_noise(
        self,
        apply_beamstop_mask: bool = True,
        apply_detector_threshold: bool = True,
        store_no_beamstop: bool = False,
    ):
        """Sample photons, convert energy to signal and apply camera acquisition.

        Physical ideal images are photon rates, not ADU. Colors are sampled
        independently, weighted by their energy-specific gains and then summed.
        An optional nonlinear curve acts on the summed per-frame linear ADU;
        it is never applied to an exposure mean or separately to each color.

        Parameters
        ----------
        apply_beamstop_mask : bool
            Apply the configured detector-plane beamstop shadow.
        apply_detector_threshold : bool
            Apply the final ADC threshold.
        store_no_beamstop : bool
            Also save an unmasked/unclipped variant from the same event draws.

        Returns
        -------
        None
            Stores the frame-averaged acquired image in hologram_exp.
        """

        rng = np.random.default_rng(getattr(self, "noise_seed", None))
        if int(self.number_frames) != self.number_frames or self.number_frames < 1:
            raise ValueError("number_frames must be a positive integer")
        self.number_frames = int(self.number_frames)
        response = getattr(self, "nonlinear_response", None)
        if response is not None and not callable(response):
            raise TypeError("nonlinear_response must be a callable of linear ADU, or None")
        if self.number_frames > 1:
            self._acquire_frames(
                rng, apply_beamstop_mask, apply_detector_threshold, store_no_beamstop
            )
            return

        # Photon statistics are independent of electronic gain. Each color has
        # its own Poisson realization and ADU/photon before signals are summed.
        signal_adu = np.zeros_like(self.hologram_detector, dtype=float)
        for gain, expected_photons in self._expected_photon_components():
            photons = rng.poisson(expected_photons).astype(np.int64)
            signal_adu += gain * self._spread_photons(photons, rng)
        if response is not None:
            converted = np.asarray(response(signal_adu.copy()))
            if np.iscomplexobj(converted):
                raise ValueError("nonlinear_response must return real ADU")
            converted = converted.astype(float)
            try:
                converted = np.broadcast_to(converted, signal_adu.shape)
            except ValueError as error:
                raise ValueError(
                    "nonlinear_response output must match the detector image"
                ) from error
            if not np.isfinite(converted).all() or np.any(converted < 0):
                raise ValueError("nonlinear_response must return finite nonnegative ADU")
            signal_adu = converted.copy()
        artifact_seed = int(rng.integers(0, np.iinfo(np.uint32).max))

        # Draw one readout image and reuse it for the optional unmasked variant.
        readout_noise = 0.0
        if self.readout_noise_average > 0 or self.readout_noise_sigma > 0:
            readout_noise = np.random.normal(
                self.readout_noise_average,
                self.readout_noise_sigma,
                signal_adu.shape,
            )

        def finalize_detection(image, apply_mask=True, apply_threshold=True):
            """Finish one physical frame using the same readout/artifact draws.

            Parameters
            ----------
            image : ndarray
                Converted ADU before readout.
            apply_mask, apply_threshold : bool
                Apply the beamstop shadow and ADC clipping.

            Returns
            -------
            ndarray
                One acquired frame, before multi-frame averaging.
            """
            detected = np.array(image, dtype=float, copy=True)
            if apply_mask:
                detected *= 1.0 - self.beamstop.beamstop
            # These effects originate in the sensor/readout and therefore are
            # not shadowed by an upstream optical beamstop.
            detected = self._add_sensor_artifacts(
                detected, np.random.default_rng(artifact_seed)
            )

            detected += readout_noise

            # Round to integer counts and optionally cap at the camera threshold.
            detected = np.round(detected, 0)
            if apply_threshold:
                detected = np.minimum(detected, self.detector_threshold)

            # Just making sure the final product is positive.
            detected[detected < 0] = 0
            return detected

        if store_no_beamstop:
            self.hologram_exp_no_beamstop = finalize_detection(
                signal_adu,
                apply_mask=False,
                apply_threshold=False,
            )

        self.hologram_exp = finalize_detection(
            signal_adu,
            apply_mask=apply_beamstop_mask,
            apply_threshold=apply_detector_threshold,
        )

    def gnomonic_projection(
        self,
        use_pixel_footprint: bool = False,
        pixel_footprint_samples: int = 3,
        ignore_flat_detector_curvature: bool = False,
    ) -> NDArray[np.float64]:
        """Project the ideal hologram onto a flat detector.

        The projection maps detector coordinates onto the simulated reciprocal
        grid and always applies the flat-pixel solid-angle collection factor,
        ``(z / sqrt(x**2 + y**2 + z**2))**3``, relative to the on-axis pixel.

        Parameters
        ----------
        use_pixel_footprint : bool, optional
            If ``True``, average the projected hologram over a regular
            sub-sampling grid inside each detector pixel. If ``False``, sample
            only at the detector pixel centre. Default is ``False``.
        pixel_footprint_samples : int, optional
            Number of sub-samples per detector-pixel axis when
            ``use_pixel_footprint`` is ``True``. A value of ``3`` uses nine
            sub-samples per detector pixel. Default is ``3``.
        ignore_flat_detector_curvature : bool, optional
            If ``True``, map detector-plane position linearly to reciprocal
            coordinates as ``qx = k * x / z`` and ``qy = k * y / z``. If
            ``False``, use the flat-detector angular mapping
            ``q = k * sin(arctan(r / z))``. Default is ``False``.

        Returns
        -------
            hologram_gnomonic : ndarray of shape (Ny, Nx)
            Gnomonic-projected hologram.
        """

        # Full reciprocal-space span of the shifted FFT grid. One FFT pixel is 1/self.real_space_pixel_size
        # therefore Dq / N, so q maps to q / Dq * N + N / 2.
        Dq = 2.0 * np.pi / self.real_space_pixel_size

        def detector_q(detx, dety):
            z = float(self.detector_layout.distance_sample_detector)
            if ignore_flat_detector_curvature:
                detqx = self.beam_parameters.wavevector * detx / z
                detqy = self.beam_parameters.wavevector * dety / z
                return np.broadcast_arrays(detqx, detqy)
            r = np.sqrt(detx**2 + dety**2)
            theta = np.arctan2(dety, detx)
            detqx = (
                self.beam_parameters.wavevector
                * np.sin(np.arctan(r / z))
                * np.cos(theta)
            )
            detqy = (
                self.beam_parameters.wavevector
                * np.sin(np.arctan(r / z))
                * np.sin(theta)
            )
            return detqx, detqy

        def solid_angle_factor(detx, dety):
            z = float(self.detector_layout.distance_sample_detector)
            r = np.sqrt(detx**2 + dety**2)
            return (z / np.sqrt(r**2 + z**2)) ** 3

        def sample_q(detqx, detqy):
            # Rescale detector q-coordinates into hologram pixel coordinates.
            # The hologram is an intensity, so use positivity-preserving linear
            # interpolation. Higher-order splines can ring below zero near sharp
            # hologram features even when every input pixel is non-negative.
            return map_coordinates(
                self.hologram,
                [
                    detqy / Dq * self.hologram.shape[0]
                    + 1 * self.hologram.shape[0] / 2,
                    detqx / Dq * self.hologram.shape[1]
                    + 1 * self.hologram.shape[1] / 2,
                ],
                order=1,
                mode="constant",
                cval=0,
            )

        if not use_pixel_footprint:
            if ignore_flat_detector_curvature:
                detqx, detqy = detector_q(
                    self.detector_layout.detx,
                    self.detector_layout.dety,
                )
            else:
                detqx = self.detector_layout.detqx
                detqy = self.detector_layout.detqy
            self.hologram_detector = sample_q(
                detqx,
                detqy,
            ) * solid_angle_factor(self.detector_layout.detx, self.detector_layout.dety)
            return self.hologram_detector

        n_samples = max(1, int(pixel_footprint_samples))
        offsets = (
            (np.arange(n_samples, dtype=float) + 0.5) / n_samples - 0.5
        ) * self.detector_layout.pixel_size
        accumulated = np.zeros(self.detector_layout.detector_shape, dtype=float)
        for y_offset in offsets:
            dety = self.detector_layout.dety + y_offset
            for x_offset in offsets:
                detx = self.detector_layout.detx + x_offset
                detqx, detqy = detector_q(detx, dety)
                accumulated += sample_q(detqx, detqy) * solid_angle_factor(detx, dety)

        self.hologram_detector = accumulated / n_samples**2

        return self.hologram_detector

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


def add_sensor_artifacts(
    image,
    *,
    artifacts_config=None,
    exposure_time=1.0,
    number_frames=1,
    detector_threshold=64e3,
    exposure_seed=None,
):
    """Add reproducible camera defects and transient cosmic rays to an image.

    ``camera_seed`` in ``artifacts_config`` fixes hot/cold pixel coordinates.
    ``exposure_seed`` controls flicker and cosmic rays for this image. When
    ``exposure_time`` is ``None``, cosmic-ray sampling assumes one second.

    Parameters
    ----------
    image : ndarray
        Input two-dimensional detector image.
    artifacts_config : dict or None
        Camera-defect and cosmic-ray settings.
    exposure_time : float or None
        Exposure duration in seconds; ``None`` means one second.
    number_frames : int
        Number of frames represented by the image.
    detector_threshold : float
        Default hot-pixel and cosmic-ray count value.
    exposure_seed : int or None
        Seed for transient artifacts in this exposure.

    Returns
    -------
    tuple of ndarray and dict
        Corrupted image and sampled artifact metadata.
    """
    model = object.__new__(detector_hologram)
    model.hologram = np.asarray(image)
    model.number_frames = int(number_frames)
    model.exposure_time = exposure_time
    model.detector_threshold = float(detector_threshold)
    model._apply_artifacts_config(artifacts_config)
    result = model._add_sensor_artifacts(
        np.asarray(image, dtype=float) * model.number_frames,
        np.random.default_rng(exposure_seed),
    )
    result /= model.number_frames
    metadata = {
        "hot_pixels": model.camera_defect_map["n_hot"],
        "cold_pixels": model.camera_defect_map["n_cold"],
        "cosmic_rays": model.cosmic_ray_count,
        "effective_exposure_time": 1.0 if exposure_time is None else float(exposure_time),
    }
    return result, metadata
