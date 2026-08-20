"""Hologram simulation pipeline for generating paired CR/CL datasets.

The pipeline follows the same config-object sequence as the interactive
tutorial (test.ipynb): XRayConfig → DetectorConfig → SampleConfig →
MagneticPatternConfig → FrontApertureConfig → IlluminationConfig →
SamplePropagatorConfig → HologramConfig.

Typical usage
-------------
from scattering_calculator.simulation_pipelines.pipelines import (
    HologramPipeline,
    HologramPipelineConfig,
    HologramPipelineRanges,
)
from scattering_calculator.simulation_pipelines import Uniform, Choice
import numpy as np

config = HologramPipelineConfig(
    recipe="Au(700)/Cr(300)/SiN(200)/Co(90)/Pt(120)/Al(60)",
    pattern_config={
        "angle_stripes": np.pi / 4,
        "stripe_width": 20e-9,
        "sigma": 1e-9,
        "waviness_amplitude": 20e-9,
        "waviness_scale": 20e-9,
    },
)

ranges = HologramPipelineRanges(
    xray_energy=Uniform(770, 790),
    detector_distance=Choice((0.02, 0.04)),
    pattern_config={
        "stripe_width": Uniform(10e-9, 50e-9),
        "waviness_amplitude": Uniform(5e-9, 30e-9),
    },
)

pipeline = HologramPipeline(config, ranges, "dataset.h5", n_samples=100)
pipeline.run()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from scattering_calculator.experimental_conditions import detector, light_beam
from scattering_calculator.sample_generator import pattern_generator
from scattering_calculator.sample_generator.domain_analysis import (
    classify_magnetic_domains,
)
from scattering_calculator.simulation_pipelines.simulation_configuration import (
    BeamstopConfig,
    DetectorConfig,
    FrontApertureConfig,
    HologramConfig,
    IlluminationConfig,
    MagneticPatternConfig,
    SampleConfig,
    SamplePropagatorConfig,
    XRayConfig,
)
from scattering_calculator.simulation_pipelines.simulation_configuration_range import (
    Choice,
    Uniform,
    _s,
)


def _beam_direction_for_projected_jones(
    alpha_beam: tuple[float, float] | float,
) -> np.ndarray | None:
    """Return a beam direction only when tilted contrast projection is needed."""
    direction = light_beam.beam_direction_from_alpha(alpha_beam)
    if np.allclose(direction, np.array([0.0, 0.0, 1.0]), atol=1e-14, rtol=0.0):
        return None
    return direction


@dataclass
class HologramPipelineConfig:
    """Fixed physical parameters shared across all simulation runs.

    Any parameter can be overridden or swept using :class:`HologramPipelineRanges`.
    Physical lengths are in SI units (metres, eV).

    Parameters
    ----------
    recipe : str
        Multilayer thin-film recipe, e.g. ``"Au(700)/Cr(300)/SiN(200)/Co(90)/Pt(120)/Al(60)"``.
        Thicknesses are in nanometres, ordered top-to-bottom.
        Slash-separated terms are separate layers; adjacent terms such as
        ``"Pt(4)Co(6)"`` are combined into one thickness-weighted
        effective-medium layer.
    sample_name : str or None
        Optional label stored in the HDF5 metadata.
    xray_energy : float
        Photon energy in eV.
    xray_photon_flux : float
        Photon flux in photons per pulse.
    xray_coherence_length : tuple of float
        Transverse coherence length in metres as ``(y, x)``.
    detector_shape : tuple of int
        Detector size ``(rows, cols)`` in pixels.
    detector_pixel_size : float
        Physical pixel pitch in metres.
    detector_distance : float
        Sample-to-detector distance in metres.
    detector_center : tuple of int or None
        Pixel position of the direct beam. ``None`` → ``shape // 2``.
    detector_noise_rms : float
        Deprecated alias for ``detector_params["readout_noise_sigma"]``.
        Leave as ``None`` and use the detector-parameter dictionary.
    detector_quantum_efficiency : float
        Deprecated alias for ``detector_params["quantum_efficiency"]``.
        Leave as ``None`` and use the detector-parameter dictionary.
    detector_params : dict
        Detector-response parameters. Canonical keys include
        ``readout_noise_average``, ``readout_noise_sigma``,
        ``detector_threshold``, ``counts_per_photon``, and
        ``quantum_efficiency``.
    measurement_config : dict
        Acquisition settings: ``exposure_time``, ``number_frames``, and
        ``max_counts_per_image``.
    use_detector_pixel_footprint : bool
        If ``True``, average the ideal hologram projection over a sub-sampling
        grid inside each detector pixel. This is more physical for finite
        detector pixels but slower than centre sampling.
    detector_pixel_footprint_samples : int
        Number of sub-samples per detector-pixel axis when
        ``use_detector_pixel_footprint`` is ``True``.
    ignore_flat_detector_curvature : bool
        If ``True``, project detector pixels onto reciprocal space with the
        linear approximation ``qx = k * x / z`` and ``qy = k * y / z``. The
        default ``False`` includes the flat-detector angular q distortion.
    artifacts_config : dict
        Photon-event shape and splatting settings. It does not own
        ``counts_per_photon``; that value belongs in ``detector_params``.
    beamstop_method : {"circular"} or None
        Beamstop shape. ``None`` → transparent (no beamstop).
    beamstop_distance : float
        Sample-to-beamstop distance in metres.
    beamstop_config : dict
        Optional keyword arguments forwarded to the beamstop generator. If
        ``"radius"`` is missing, no beamstop or wire is created. Beamstop
        lengths, including ``"sigma"``, are in metres.
    save_detected_hologram_without_beamstop : bool
        If ``True``, save an additional detected hologram source named
        ``detected_no_beamstop``. It is produced by the same detector-noise
        pipeline, but skips the beamstop mask and detector threshold cap before
        frame averaging. Default ``False``.
    aperture_method : {"FTH_circular"} or None
        Holography mask layout. ``None`` → fully transparent.
    aperture_types : list of {"OH", "RH", "SLIT"}
        Object hole (OH), reference hole (RH), or rectangular slit for each
        aperture.
    aperture_radii : list of float
        Aperture radii in metres. For slits, this value is the slit width.
    aperture_lengths : list of float
        Slit lengths in metres. Ignored for circular OH/RH apertures.
    aperture_centers : list of (float, float)
        Aperture centres ``(y, x)`` in metres relative to the sample centre.
    aperture_sigmas : list of float
        Edge-softening sigma in metres for each aperture.
    aperture_angles : list of float
        Ellipse orientation in radians for each aperture.
    aperture_ellipticities : list of float
        Ellipse y/x axis ratio for each aperture. ``1`` keeps a circular hole.
    aperture_roughnesses : list of float
        Random boundary roughness amplitude for each aperture.
    aperture_roughness_modes : list of (int, int)
        Inclusive Fourier-mode range used for each rough aperture boundary.
    aperture_seeds : list of int
        Random seeds used for rough aperture boundaries. Negative values mean
        no fixed seed.
    illumination_function : {"gaussian"} or None
        Spatial beam profile. ``None`` → plane wave.
    illumination_center : (float, float)
        Beam centre ``(y, x)`` in metres.
    illumination_focus_distance : float
        Propagation distance from the Gaussian waist to the sample plane, in metres.
    illumination_fwhm : float
        Gaussian beam FWHM at the waist in metres.
    illumination_alpha_beam : (float, float)
        Beam tilt ``(alpha_y, alpha_x)`` in radians. ``(0, 0)`` keeps normal
        incidence. A scalar value is accepted as an x-direction tilt for
        backward compatibility.
    pattern_type : {"wavy_stripe_pattern", "binary_labyrinth_pattern", "disordered_skyrmion_lattice_pattern", "saturated_pattern", "skyrmion_pattern", "image_pattern"}
        Which magnetic domain pattern generator to use.
    pattern_config : dict
        Pattern parameters forwarded to the generator. Physical-length entries
        are specified in metres and converted to pixels by
        :class:`MagneticPatternConfig` for the selected pattern type
        (e.g. ``{"angle_stripes": np.pi/4, "stripe_width": 20e-9,
        "sigma": 1e-9, "waviness_amplitude": 20e-9,
        "waviness_scale": 20e-9}``).
    pattern_config_length : dict
        Deprecated compatibility dict for physical-length pattern parameters in
        metres. Prefer putting these values directly in ``pattern_config``.
    use_roi : bool
        Master switch for ROI accelerations. If ``True``, use local aperture
        regions for magnetic-pattern generation, aperture-mask generation,
        dielectric tensor construction, and Jones propagation. If ``False``,
        compute those steps on the full sample plane. Default ``True``.
    magnetic_pattern_use_roi : bool
        If ``True``, generate wavy stripe patterns only in a padded ROI around
        the object hole and paste that ROI into a uniform full-field pattern.
        If ``False``, generate the magnetic pattern over the entire sample
        plane. Default ``True``.
    magnetic_pattern_classification_min_area : int
        Minimum connected-component area in object-hole pixels for a resolved
        stripe or bubble. Smaller components are counted as noise.
    magnetic_pattern_bubble_max_eccentricity : float
        Maximum covariance eccentricity for a component to count as a bubble.
    magnetic_pattern_bubble_min_circularity : float
        Minimum digital circularity for a component to count as a bubble.
    dielectric_tensor_use_roi : bool
        If ``True``, compute magnetic/vacuum dielectric-tensor corrections in
        local aperture bounding boxes. If ``False``, use the full aperture
        support mask, matching the pre-ROI tensor path for timing comparisons.
        Default ``True``.
    dielectric_tensor_compact : bool
        If ``True``, keep the dielectric tensor as constant per-layer diagonal
        terms plus aperture ROI patches and evaluate those patches during Jones
        propagation. This avoids allocating the full dense tensor stack. Default
        ``True``.
    propagate : bool
        If ``True``, propagate the Jones field through free space between
        material layers using each layer thickness and the sample pixel size.
        If ``False``, skip transverse FFT diffraction and use the
        ``jones_apply_zero_order_phase`` option for the longitudinal phase.
        Default ``False``.
    jones_apply_zero_order_phase : bool
        If ``True`` and ``propagate=False``, multiply the Jones field by the
        zero-spatial-frequency free-space phase ``exp(-1j * k0 * dz)`` between
        material slices. This is the default Jones-only approximation because
        it preserves the layer-to-layer longitudinal phase without paying for
        multislice FFT propagation. Set ``False`` only when reproducing older
        Jones-only results.
    propagation_padding_px : int
        Number of pixels to pad on each side during each free-space
        propagation step. Padding is cropped away after propagation and reduces
        periodic FFT wraparound. ``0`` disables padding. Default ``0``.
    propagation_padding_mode : str
        NumPy padding mode for free-space propagation margins. ``"edge"`` and
        ``"reflect"`` avoid the hard crop-to-zero discontinuity of
        ``"constant"`` padding. Default ``"edge"``.
    propagation_absorber_width_px : int
        Width of the smooth edge absorber used during free-space propagation.
        When padding is enabled, the absorber is clamped to the padded margin so
        it does not damp the returned crop. ``0`` disables absorption. Default
        ``0``.
    propagation_absorber_strength : float
        Strength of the exponential edge absorber. Larger values damp the edge
        more strongly. Default ``0``.
    propagation_absorber_profile : str
        Smooth profile used to ramp absorption from zero in the interior to the
        configured strength at the outer edge. One of ``"cosine"``,
        ``"smoothstep"``, ``"quadratic"``, or ``"linear"``. Default
        ``"cosine"``.
    multislice_propagation_roi : bool
        If ``True``, approximate the free-space step between slices by running
        angular-spectrum propagation only inside aperture ROI boxes. The full
        field starts from the plane-wave phase advance; each ROI crop then adds
        its local diffraction correction relative to that baseline, so
        ROI boxes add corrections without overwriting one another. If
        ``False``, use full-field free-space propagation. Default ``False``.
    multislice_propagation_roi_padding_px : int
        Extra pixels added around each aperture ROI box before approximate
        ROI-only free-space propagation. This is separate from
        ``propagation_padding_px``, which pads FFT boundaries inside each ROI
        crop but does not enlarge the returned propagated area. The local
        diffraction correction is smoothly tapered across this ROI padding
        before it is added back to the full field, which avoids hard rectangular
        paste edges. Default ``0``.
    multislice_propagation_roi_merge_overlaps : bool
        User-facing ROI merge preference. If ``True``, all padded aperture
        support boxes are enclosed in one common multislice propagation crop.
        If ``False``, disjoint padded boxes from physically separate aperture
        supports may remain separate for speed. Aperture supports that overlap
        in the physical material mask are always merged before padding,
        regardless of this flag, because intersecting funnels are one hole
        system. Padded boxes that overlap are also merged even if this is
        ``False``; otherwise the same pixels would receive multiple local
        diffraction corrections. Default ``True``.
    farfield_oversampling : int
        Factor used to extend the complex exit wave before the far-field FFT.
        Values above ``1`` fill the larger field with the configured
        illumination multiplied by the uniform background stack and ROI
        free-space factors, then paste the simulated central exit wave into the
        middle. Default ``1``.
    propagator_method : {"Jones", "Scalar", "Stokes"}
        Light-matter interaction method. ``"Jones"`` uses the full
        two-component Jones matrix formalism. ``"Scalar"`` skips dielectric
        tensor construction, builds a complex refractive-index stack directly
        from the database channels ``[n_total, n_circ, n_lin]``, and propagates
        one scalar eigenmode. It is faster and matches Jones propagation when
        the configured polarization is an eigenvector of the local interaction.
        Default ``"Jones"``.
    scalar_refractive_index_lazy : bool
        Scalar-mode memory option. If ``True``, compute scalar aperture ROI
        patches layer by layer during propagation instead of precomputing all
        scalar patches for the full stack. This mirrors the compact Jones
        memory-saving path while avoiding dielectric tensor allocation.
        Default ``True``.
    oversampling : int
        Oversampling factor relative to the Nyquist limit from the detector.
        ``real_space_pixel_size = detector_resolution / oversampling``.
        Default 2.
    random_seed : int or None
        Seed for reproducible sweep sampling and per-sample stochastic effects.
        ``None`` keeps stochastic behavior non-deterministic. Default ``None``.
    """

    recipe: str
    sample_name: str | None = None

    # X-ray source
    xray_energy: float = 787.9  # eV
    xray_photon_flux: float = 1e12  # photons/pulse
    xray_coherence_length: tuple[float, float] = (100e-6, 100e-6)  # m, (y, x)

    # Detector
    detector_shape: tuple[int, int] = (1300, 1300)  # px
    detector_pixel_size: float = 20e-6  # m/px
    detector_distance: float = 0.02  # m
    detector_center: tuple[int, int] | None = None  # None → shape // 2
    detector_noise_rms: float | None = None
    detector_quantum_efficiency: float | None = None
    detector_params: dict = field(
        default_factory=lambda: {
            "readout_noise_average": 50,
            "detector_threshold": 64e3,
            "counts_per_photon": 100,
        }
    )
    artifacts_config: dict = field(
        default_factory=lambda: {
            "sigma_photon": 0.75,
            "photon_n_classes": 1,
            "photon_n_variants": 30,
            "photon_kernel_size": 9,
            "photon_irregularity": 2.0,
            "regenerate_photon_kernels": True,
        }
    )
    measurement_config: dict = field(
        default_factory=lambda: {
            "exposure_time": 1.0,
            "number_frames": 1,
            "max_counts_per_image": None,
        }
    )
    use_detector_pixel_footprint: bool = False
    detector_pixel_footprint_samples: int = 3
    ignore_flat_detector_curvature: bool = False

    # Beamstop
    beamstop_method: str | None = "circular"
    beamstop_distance: float = 0.001  # m
    beamstop_config: dict = field(default_factory=dict)
    save_detected_hologram_without_beamstop: bool = False

    # FTH holography mask
    aperture_method: str | None = "FTH_circular"
    aperture_types: list[str] = field(
        default_factory=lambda: ["OH", "RH", "RH"]
    )
    aperture_radii: list[float] = field(
        default_factory=lambda: [60e-9, 6e-9, 4e-9]
    )  # m
    aperture_lengths: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )  # m, only used by SLIT apertures
    aperture_depths: list[float | None] | None = None  # m, None uses defaults
    aperture_centers: list[tuple] = field(
        default_factory=lambda: [
            (0.0, 0.0),
            (0.2e-6, -0.15e-6),
            (0.15e-6, 0.15e-6),
        ]
    )  # m, (y, x)
    aperture_sigmas: list[float] = field(
        default_factory=lambda: [1e-9, 2e-9, 2e-9]
    )  # m
    aperture_angles: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )  # rad
    aperture_ellipticities: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
    )
    aperture_roughnesses: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    aperture_roughness_modes: list[tuple[int, int]] = field(
        default_factory=lambda: [(0, 0), (0, 0), (0, 0)]
    )
    aperture_seeds: list[int] = field(default_factory=lambda: [-1, -1, -1])
    aperture_top_radius_factors: list[float] = field(
        default_factory=lambda: [2.0, 2.0, 2.0]
    )

    # Illumination
    illumination_function: str | None = "gaussian"
    illumination_center: tuple[float, float] = (0.0, 0.0)  # m
    illumination_focus_distance: float = 1e-3  # m
    illumination_fwhm: float = 0.5e-6  # m
    illumination_alpha_beam: tuple[float, float] | float = (0.0, 0.0)  # rad, (alpha_y, alpha_x)

    # Magnetic domain pattern
    pattern_type: str = "wavy_stripe_pattern"
    pattern_config: dict = field(default_factory=dict)
    pattern_config_length: dict = field(default_factory=dict)
    use_roi: bool = True
    magnetic_pattern_use_roi: bool = True
    magnetic_pattern_classification_min_area: int = 9
    magnetic_pattern_bubble_max_eccentricity: float = 0.85
    magnetic_pattern_bubble_min_circularity: float = 0.45
    dielectric_tensor_use_roi: bool = True
    dielectric_tensor_compact: bool = True
    dielectric_tensor_local_k_projection: bool = False
    propagate: bool = False
    jones_apply_zero_order_phase: bool = True
    propagation_padding_px: int = 0
    propagation_padding_mode: str = "edge"
    propagation_absorber_width_px: int = 0
    propagation_absorber_strength: float = 0.0
    propagation_absorber_profile: str = "cosine"
    multislice_propagation_roi: bool = False
    multislice_propagation_roi_padding_px: int = 0
    multislice_propagation_roi_merge_overlaps: bool = True
    farfield_oversampling: int = 1
    propagator_method: str = "Jones"
    scalar_refractive_index_lazy: bool = True
    sample_tilt_theta: float = 0.0  # rad
    sample_tilt_axis: str = "x"
    sample_tilt_voxel_size: float | None = None  # m
    sample_tilt_antialias_samples: int = 3

    # Simulation grid
    oversampling: int = 2
    random_seed: int | None = None


@dataclass
class HologramPipelineRanges:
    """Parameter distributions for sweeping across simulation runs.

    Set a field to:

    * ``None`` — use the fixed value from :class:`HologramPipelineConfig`.
    * A scalar — override with that fixed value for every run.
    * :class:`Uniform` ``(low, high)`` — sample uniformly each run.
    * :class:`Choice` ``((a, b, ...))`` — sample from a discrete set each run.

    For ``pattern_config``, individual dict values can themselves be
    :class:`Uniform` or :class:`Choice` samplers; keys not listed in the range
    dict fall back to the base config value. The range dict may also be a
    callable that returns a fully sampled dict for interdependent parameters.
    ``pattern_config_length`` is still accepted for older callers.
    ``beamstop_config`` follows the same pattern.

    Examples
    --------
    >>> ranges = HologramPipelineRanges(
    ...     xray_energy=Uniform(770, 790),
    ...     detector_distance=Choice((0.02, 0.04, 0.08)),
    ...     pattern_config={"stripe_width": Uniform(10e-9, 50e-9)},
    ... )
    """

    # X-ray
    xray_energy: float | Uniform | None = None
    xray_photon_flux: float | Uniform | None = None
    xray_coherence_length: (
        tuple[float, float]
        | tuple[Uniform, Uniform]
        | Uniform
        | Callable[[dict[str, Any]], tuple[float, float]]
        | None
    ) = None

    # Detector
    detector_distance: float | Uniform | Callable[[dict[str, Any]], Any] | None = None
    detector_pixel_size: float | Uniform | None = None
    detector_center: (
        tuple[float | Uniform, float | Uniform]
        | Callable[[dict[str, Any]], tuple[float, float]]
        | None
    ) = None
    detector_noise_rms: float | Uniform | None = None
    detector_params: dict | None = None
    artifacts_config: dict | None = None
    measurement_config: dict | None = None
    use_detector_pixel_footprint: bool | None = None
    detector_pixel_footprint_samples: int | Uniform | None = None
    ignore_flat_detector_curvature: bool | None = None

    # Beamstop
    beamstop_config: dict | None = None

    # FTH holography mask
    aperture_config: dict | Callable[[dict[str, Any]], dict] | None = None

    # Magnetic pattern
    pattern_type: str | Choice | None = None
    pattern_config: dict | Callable[[dict[str, Any]], dict] | None = None
    pattern_config_length: dict | None = None

    # Illumination
    illumination_focus_distance: float | Uniform | None = None
    illumination_fwhm: float | Uniform | None = None
    illumination_alpha_beam: (
        tuple[float | Uniform, float | Uniform]
        | Callable[[dict[str, Any]], tuple[float, float]]
        | float
        | Uniform
        | None
    ) = None
    illumination_center: (
        tuple[float | Uniform, float | Uniform]
        | Callable[[dict[str, Any]], tuple[float, float]]
        | None
    ) = None
    dielectric_tensor_local_k_projection: bool | Choice | None = None


class HologramPipeline:
    """Generate paired CR/CL holograms for *n_samples* parameter configurations.

    For each configuration the pipeline builds and calls the full chain of
    simulation_configuration config objects in the same order as the
    interactive tutorial (test.ipynb):

    1. :class:`XRayConfig` — photon energy, flux, coherence
    2. :class:`BeamstopConfig` + :class:`DetectorConfig` — detector geometry
       and derived sample-plane pixel size
    3. :class:`SampleConfig` — multilayer structure at the derived resolution
    4. :class:`MagneticPatternConfig` — domain pattern generation
    5. :class:`FrontApertureConfig` — FTH holography mask
    6. :class:`IlluminationConfig` — beam profile
    7. :class:`SamplePropagatorConfig` + :class:`HologramConfig` — Jones
       propagation and hologram accumulation for CR and CL

    HDF5 output layout
    ------------------
    ::

        output.h5
        ├── _pipeline_config/       ← file-global n_samples and oversampling
        ├── 00000/
        │   ├── CR/
        │   │   ├── ideal           ← float32 2D hologram
        │   │   ├── detected        ← float32 2D hologram
        │   │   └── exit_wave       ← complex64 2D wavefield
        │   ├── CL/
        │   │   └── ...
        │   ├── beamstop_mask       ← 2D boolean mask
        │   └── metadata/           ← all scalar parameters as datasets
        │       ├── xray/
        │       ├── detector/
        │       └── sample/
        │           ├── magnetic_pattern/
        │           └── aperture/
        │               └── aperture_config/
        │               ├── apertures_type    ← aperture labels, e.g. OH/RH/SLIT
        │               ├── apertures_radius  ← radii, or slit widths, in metres
        │               ├── apertures_length  ← slit lengths in metres
        │               ├── apertures_center  ← aperture centres in metres, (y, x)
        │               ├── apertures_sigma   ← aperture edge sigmas in metres
        │               ├── apertures_angle   ← ellipse angles in radians
        │               ├── apertures_ellipticity
        │               ├── apertures_roughness
        │               ├── apertures_roughness_modes
        │               └── apertures_seed
        ├── 00001/
        │   └── ...

    Parameters
    ----------
    config : HologramPipelineConfig
        Fixed parameters shared across all runs.
    ranges : HologramPipelineRanges
        Distributions for parameters that vary between runs.
    output_path : Path or str
        Output HDF5 file path. Created (or overwritten) when ``run()`` is called.
    n_samples : int
        Number of parameter configurations to simulate.
    verbose : bool
        Print per-sample timing to stdout. Default ``True``.
    """

    _POLARIZATIONS = ("CR", "CL")

    def __init__(
        self,
        config: HologramPipelineConfig,
        ranges: HologramPipelineRanges,
        output_path: Path | str,
        n_samples: int,
        verbose: bool = True,
    ) -> None:
        """Store pipeline configuration and output settings.

        Parameters
        ----------
        config : HologramPipelineConfig
            Fixed parameters shared by every simulated sample.
        ranges : HologramPipelineRanges
            Optional parameter distributions or overrides sampled per run.
        output_path : Path or str
            HDF5 file path written by :meth:`run`.
        n_samples : int
            Number of simulated configurations to generate.
        verbose : bool
            If ``True``, print progress and timing information.

        Returns
        -------
        None
            The instance is initialised in place.
        """
        self.config = config
        self.ranges = ranges
        self.output_path = Path(output_path)
        self.n_samples = n_samples
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the simulation loop and write all results to HDF5.

        Parameters
        ----------
        None
            This method uses the pipeline configuration stored on ``self``.

        Returns
        -------
        None
            Results are written to ``self.output_path``.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        width = len(str(self.n_samples))
        t_start = time.time()
        previous_random_state = None
        if self.config.random_seed is not None:
            previous_random_state = np.random.get_state()
            np.random.seed(self.config.random_seed)
        try:
            with h5py.File(self.output_path, "w") as h5:
                self._write_pipeline_config(h5)
                for i in range(self.n_samples):
                    if self.verbose:
                        print(
                            f"  [{i + 1:>{width}}/{self.n_samples}] Simulating...",
                            end=" ",
                            flush=True,
                        )
                    t0 = time.time()
                    self._simulate_one(h5, i)
                    if self.verbose:
                        print(f"done ({time.time() - t0:.1f} s)")
        finally:
            if previous_random_state is not None:
                np.random.set_state(previous_random_state)
        if self.verbose:
            elapsed = time.time() - t_start
            print(
                f"Pipeline complete: {self.n_samples} configurations in "
                f"{elapsed:.1f} s → {self.output_path}"
            )

    def write_precomputed_result(
        self,
        hologram_config: HologramConfig,
        detector_config: DetectorConfig,
        metadata: dict[str, Any],
        aperture_config: dict[str, Any],
        supportmask: np.ndarray,
        magnetic_pattern_oh: np.ndarray,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Write one already-simulated result using the canonical pipeline layout.

        This is useful for interactive notebooks that construct each simulation
        object explicitly but should produce HDF5 files compatible with
        :meth:`run`.

        Parameters
        ----------
        hologram_config : HologramConfig
            Container with CR/CL exit waves and holograms.
        detector_config : DetectorConfig
            Configured detector containing the detector layout and beamstop.
        metadata : dict[str, Any]
            Canonical flat metadata paths for the simulated result.
        aperture_config : dict[str, Any]
            Pipeline-style aperture configuration with ``aperture_*`` keys.
        supportmask : np.ndarray
            Detector-space aperture support mask.
        magnetic_pattern_oh : np.ndarray
            Magnetic pattern visible through the object hole.
        overwrite : bool, optional
            Replace an existing output file when ``True``.

        Returns
        -------
        pathlib.Path
            Path of the written HDF5 file.
        """
        if self.n_samples != 1:
            raise ValueError(
                "write_precomputed_result writes exactly one sample; "
                "construct the pipeline with n_samples=1."
            )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "x"
        with h5py.File(self.output_path, mode) as h5:
            self._write_pipeline_config(h5)
            self._write_sample(
                h5,
                0,
                hologram_config,
                detector_config,
                metadata,
                aperture_config=aperture_config,
                supportmask=supportmask,
                magnetic_pattern_oh=magnetic_pattern_oh,
            )
        return self.output_path

    @classmethod
    def build_precomputed_metadata(
        cls,
        *,
        xray_config: XRayConfig,
        detector_config: DetectorConfig,
        beamstop_config: BeamstopConfig,
        sample_config: SampleConfig,
        magnetic_pattern_config: MagneticPatternConfig,
        aperture_config: FrontApertureConfig,
        illumination_config: IlluminationConfig,
        propagator_config: SamplePropagatorConfig,
        use_roi: bool,
        magnetic_pattern_use_roi: bool,
        dielectric_tensor_use_roi: bool,
        dielectric_tensor_compact: bool,
        save_detected_hologram_without_beamstop: bool,
        dielectric_tensor_local_k_projection: bool = False,
    ) -> dict[str, Any]:
        """Build canonical metadata for an interactively simulated result."""
        metadata: dict[str, Any] = {}
        metadata.update(beamstop_config.get_metadata(prefix="beamstop/"))
        metadata.update(cls._detector_metadata(detector_config))
        metadata.update(sample_config.get_metadata(prefix="sample/"))

        magnetic_metadata = magnetic_pattern_config.get_metadata(
            prefix="sample/magnetic_pattern/"
        )
        magnetic_metadata.pop(
            "sample/magnetic_pattern/real_space_pixel_size", None
        )
        metadata.update(magnetic_metadata)

        aperture_metadata = aperture_config.get_metadata(prefix="sample/aperture/")
        aperture_metadata.pop("sample/aperture/real_space_pixel_size", None)
        aperture_metadata.pop("sample/aperture/use_roi", None)
        metadata.update(aperture_metadata)

        illumination_values = illumination_config.illumination_config
        metadata["illumination/function"] = str(
            illumination_config.illumination_function
        )
        metadata["illumination/center_m"] = np.asarray(
            illumination_values.get("center", (0.0, 0.0)), dtype=float
        )
        metadata["illumination/focus_distance_m"] = illumination_values.get(
            "distance", 0.0
        )
        metadata["illumination/fwhm_m"] = illumination_values.get("fwhm", 0.0)
        metadata["illumination/alpha_beam_rad"] = illumination_values.get(
            "alpha_beam", 0.0
        )

        metadata["sample/use_roi"] = bool(use_roi)
        metadata["sample/magnetic_pattern/use_roi"] = bool(
            use_roi and magnetic_pattern_use_roi
        )
        beam_direction = None
        if getattr(propagator_config, "propagator_method", None) in ("Jones", "Stokes"):
            beam_direction = _beam_direction_for_projected_jones(
                illumination_values.get("alpha_beam", 0.0)
            )
        metadata["sample/dielectric_tensor/use_roi"] = bool(
            (use_roi and dielectric_tensor_use_roi)
            if beam_direction is None and not dielectric_tensor_local_k_projection
            else False
        )
        metadata["sample/dielectric_tensor/compact"] = bool(
            dielectric_tensor_compact
            if beam_direction is None and not dielectric_tensor_local_k_projection
            else False
        )
        metadata["sample/dielectric_tensor/beam_direction_projected"] = (
            beam_direction is not None
        )
        metadata["sample/dielectric_tensor/local_k_projected"] = bool(
            dielectric_tensor_local_k_projection
        )
        if beam_direction is not None:
            metadata["sample/dielectric_tensor/beam_direction_xyz"] = beam_direction
        metadata.update(cls._propagator_metadata(propagator_config))
        metadata.update(cls._xray_metadata(xray_config))
        metadata["detector/save_detected_no_beamstop"] = bool(
            save_detected_hologram_without_beamstop
        )
        return metadata

    @staticmethod
    def _propagator_metadata(
        propagator_config: SamplePropagatorConfig,
    ) -> dict[str, Any]:
        """Return propagation metadata grouped under ``propagator_config/``."""
        values: dict[str, Any] = {
            "propagator_method": str(propagator_config.propagator_method)
        }
        values.update(
            {
                key: value
                for key, value in propagator_config.propagator_config.items()
                if key != "propagator_method"
            }
        )
        return {f"propagator_config/{key}": value for key, value in values.items()}

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def _sample_params(self) -> dict[str, Any]:
        """Draw one sampled parameter set, merging ranges over the base config.

        Parameters
        ----------
        None
            This method samples from ``self.config`` and ``self.ranges``.

        Returns
        -------
        params : dict[str, Any]
            Fully resolved parameters for one simulated sample.
        """
        cfg = self.config
        rng = self.ranges

        def _resolve(value: Any, params: dict[str, Any]) -> Any:
            """Resolve callables, tuples, and sampler objects against parameters.

            Parameters
            ----------
            value : Any
                Fixed value, callable, tuple, or sampler to resolve.
            params : dict[str, Any]
                Parameters already sampled for the current run.

            Returns
            -------
            resolved : Any
                Concrete value to store in the sampled parameter dict.
            """
            if callable(value):
                value = value(params)
            if isinstance(value, tuple):
                return tuple(_resolve(v, params) for v in value)
            return _s(value)

        def _pick(range_val: Any, base_val: Any, params: dict[str, Any]) -> Any:
            """Return a sampled range value or the fixed base value.

            Parameters
            ----------
            range_val : Any
                Optional range override from :class:`HologramPipelineRanges`.
            base_val : Any
                Fixed fallback value from :class:`HologramPipelineConfig`.
            params : dict[str, Any]
                Parameters already sampled for the current run.

            Returns
            -------
            value : Any
                ``base_val`` when ``range_val`` is ``None``; otherwise the
                resolved range value.
            """
            return base_val if range_val is None else _resolve(range_val, params)

        def _sample_dict_with_params(d: dict, params: dict[str, Any]) -> dict:
            """Sample each value in a nested configuration dictionary.

            Parameters
            ----------
            d : dict
                Dictionary whose values may be fixed values, samplers,
                callables, or nested dictionaries.
            params : dict[str, Any]
                Parameters already sampled for the current run.

            Returns
            -------
            sampled : dict
                Dictionary with all supported values resolved.
            """
            return {
                k: (
                    _sample_dict_with_params(v, params)
                    if isinstance(v, dict)
                    else _resolve(v, params)
                )
                for k, v in d.items()
            }

        def _merge_dict(
            range_dict: dict | None, base_dict: dict, params: dict[str, Any]
        ) -> dict:
            """Merge sampled dictionary overrides over fixed defaults.

            Parameters
            ----------
            range_dict : dict or None
                Optional override dictionary from the sweep ranges.
            base_dict : dict
                Fixed dictionary from the base configuration.
            params : dict[str, Any]
                Parameters already sampled for the current run.

            Returns
            -------
            merged : dict
                Sampled override values merged over the fixed defaults.
            """
            if range_dict is not None:
                if callable(range_dict):
                    resolved = _resolve(range_dict, params)
                    return _sample_dict_with_params(resolved, params)
                merged = dict(base_dict)
                merged.update(_sample_dict_with_params(range_dict, params))
                return merged
            merged = dict(base_dict)
            return merged

        params: dict[str, Any] = {
            "detector_shape": cfg.detector_shape,
            "detector_center": cfg.detector_center,
            "oversampling": cfg.oversampling,
        }
        params["energy"] = _pick(rng.xray_energy, cfg.xray_energy, params)
        params["photon_flux"] = _pick(
            rng.xray_photon_flux, cfg.xray_photon_flux, params
        )
        params["coherence_length"] = _pick(
            rng.xray_coherence_length, cfg.xray_coherence_length, params
        )
        params["detector_center"] = _pick(
            rng.detector_center, cfg.detector_center, params
        )
        params["detector_pixel_size"] = _pick(
            rng.detector_pixel_size, cfg.detector_pixel_size, params
        )
        params["detector_params"] = _merge_dict(
            rng.detector_params, cfg.detector_params, params
        )
        params["detector_params"] = self._normalize_detector_params(
            params["detector_params"],
            legacy_noise_rms=_pick(
                rng.detector_noise_rms, cfg.detector_noise_rms, params
            ),
            legacy_quantum_efficiency=cfg.detector_quantum_efficiency,
        )
        params["detector_params"].setdefault("readout_noise_sigma", 3.0)
        params["detector_params"].setdefault("quantum_efficiency", 1.0)
        params["artifacts_config"] = _merge_dict(
            rng.artifacts_config, cfg.artifacts_config, params
        )
        legacy_counts_per_photon = params["artifacts_config"].pop(
            "counts_per_photon", None
        )
        detector_counts_per_photon = params["detector_params"].get(
            "counts_per_photon"
        )
        if legacy_counts_per_photon is not None:
            if (
                detector_counts_per_photon is not None
                and detector_counts_per_photon != legacy_counts_per_photon
            ):
                raise ValueError(
                    "Conflicting counts_per_photon values: use only "
                    "detector_params['counts_per_photon']."
                )
            params["detector_params"]["counts_per_photon"] = (
                legacy_counts_per_photon
            )
        params["measurement_config"] = _merge_dict(
            rng.measurement_config, cfg.measurement_config, params
        )
        params["use_detector_pixel_footprint"] = bool(
            _pick(
                rng.use_detector_pixel_footprint,
                cfg.use_detector_pixel_footprint,
                params,
            )
        )
        params["detector_pixel_footprint_samples"] = int(
            _pick(
                rng.detector_pixel_footprint_samples,
                cfg.detector_pixel_footprint_samples,
                params,
            )
        )
        params["ignore_flat_detector_curvature"] = bool(
            _pick(
                rng.ignore_flat_detector_curvature,
                cfg.ignore_flat_detector_curvature,
                params,
            )
        )
        params["pattern_type"] = _pick(rng.pattern_type, cfg.pattern_type, params)
        params["pattern_config"] = _merge_dict(
            rng.pattern_config, cfg.pattern_config, params
        )
        params["pattern_config_length"] = _merge_dict(
            rng.pattern_config_length, cfg.pattern_config_length, params
        )
        for key, value in params["pattern_config_length"].items():
            if (
                key in params["pattern_config"]
                and not np.array_equal(params["pattern_config"][key], value)
            ):
                raise ValueError(
                    f"Conflicting magnetic-pattern value for {key!r}: use only "
                    "pattern_config. pattern_config_length is a legacy alias."
                )
            params["pattern_config"][key] = value
        params["pattern_config_length"] = {}
        if cfg.random_seed is not None:
            def _set_seed_if_missing(config_dict: dict, key: str) -> None:
                """Insert a random seed into a config dictionary when absent.

                Parameters
                ----------
                config_dict : dict
                    Configuration dictionary that may need a seed entry.
                key : str
                    Name of the seed field to populate.

                Returns
                -------
                None
                    ``config_dict`` is modified in place when ``key`` is absent.
                """
                if config_dict.get(key) is None:
                    config_dict[key] = int(np.random.randint(0, 2**31 - 1))

            params["sample_seed"] = int(np.random.randint(0, 2**31 - 1))
            _set_seed_if_missing(params["pattern_config"], "seed")
        params["detector_distance"] = _pick(
            rng.detector_distance, cfg.detector_distance, params
        )
        params["illumination_focus_distance"] = _pick(
            rng.illumination_focus_distance,
            cfg.illumination_focus_distance,
            params,
        )
        params["illumination_fwhm"] = _pick(
            rng.illumination_fwhm, cfg.illumination_fwhm, params
        )
        params["illumination_alpha_beam"] = _pick(
            rng.illumination_alpha_beam, cfg.illumination_alpha_beam, params
        )
        params["illumination_center"] = _pick(
            rng.illumination_center, cfg.illumination_center, params
        )
        params["dielectric_tensor_local_k_projection"] = bool(
            _pick(
                rng.dielectric_tensor_local_k_projection,
                cfg.dielectric_tensor_local_k_projection,
                params,
            )
        )
        params["beamstop_config"] = _merge_dict(
            rng.beamstop_config, cfg.beamstop_config, params
        )
        params["aperture_config"] = _merge_dict(
            rng.aperture_config,
            {
                "aperture_types": cfg.aperture_types,
                "aperture_radii": cfg.aperture_radii,
                "aperture_depths": cfg.aperture_depths,
                "aperture_lengths": cfg.aperture_lengths,
                "aperture_centers": cfg.aperture_centers,
                "aperture_sigmas": cfg.aperture_sigmas,
                "aperture_angles": cfg.aperture_angles,
                "aperture_ellipticities": cfg.aperture_ellipticities,
                "aperture_roughnesses": cfg.aperture_roughnesses,
                "aperture_roughness_modes": cfg.aperture_roughness_modes,
                "aperture_seeds": cfg.aperture_seeds,
                "aperture_top_radius_factors": cfg.aperture_top_radius_factors,
            },
            params,
        )
        if cfg.random_seed is not None:
            _set_seed_if_missing(params["beamstop_config"], "seed")
            _set_seed_if_missing(params["artifacts_config"], "photon_class_seed")
            _set_seed_if_missing(params["artifacts_config"], "photon_kernel_seed")
            _set_seed_if_missing(params["detector_params"], "noise_seed")
        return params

    @staticmethod
    def _normalize_detector_params(
        detector_params: dict,
        *,
        legacy_noise_rms: float | None,
        legacy_quantum_efficiency: float | None,
    ) -> dict:
        """Normalize legacy detector aliases and reject conflicting values."""
        normalized = detector.detector_hologram._normalize_detector_param_aliases(
            detector_params
        )
        for legacy_name, legacy_value, canonical_name in (
            ("detector_noise_rms", legacy_noise_rms, "readout_noise_sigma"),
            (
                "detector_quantum_efficiency",
                legacy_quantum_efficiency,
                "quantum_efficiency",
            ),
        ):
            if legacy_value is None:
                continue
            canonical_value = normalized.get(canonical_name)
            if canonical_value is not None and canonical_value != legacy_value:
                raise ValueError(
                    f"Conflicting {legacy_name} and "
                    f"detector_params['{canonical_name}']; use only the "
                    "detector_params value."
                )
            normalized[canonical_name] = legacy_value
        return normalized

    @staticmethod
    def _magnetic_pattern_roi(
        sample_shape: tuple[int, int] | np.ndarray,
        pixel_size: float,
        aperture_config: dict[str, Any],
        pattern_type: str,
        *,
        enabled: bool = True,
    ) -> tuple[tuple[int, int], tuple[slice, slice] | None, tuple[float, float] | None]:
        """Return the shape and insertion slices for an OH-local pattern.

        Wavy stripe generation is expensive on the full oversampled sample
        plane, while the magnetic texture is only visible through the OH. For
        stripe patterns, generate a padded bounding box around the OH and paste
        it into a full field initialised to +1. Other pattern generators keep
        their historical full-field behaviour.

        Parameters
        ----------
        sample_shape : tuple[int, int] or np.ndarray
            Full lateral sample shape ``(Ny, Nx)`` in pixels.
        pixel_size : float
            Real-space sample pixel size in metres.
        aperture_config : dict[str, Any]
            Aperture configuration containing aperture types, radii, centres,
            edge sigmas, roughnesses, and top radius factors.
        pattern_type : str
            Magnetic-pattern generator name.
        enabled : bool
            If ``False``, skip ROI selection and return the full sample shape.

        Returns
        -------
        roi_shape : tuple[int, int]
            Shape used to generate the magnetic pattern.
        slices : tuple[slice, slice] or None
            Insertion slices into the full magnetic-pattern array, or ``None``
            when the full sample shape is used.
        coordinate_offset : tuple[float, float] or None
            ROI centre offset from the full sample centre in pixels, or
            ``None`` when the full sample shape is used.
        """
        full_shape = tuple(int(v) for v in sample_shape)
        roi_pattern_types = {
            "wavy_stripe_pattern",
            "binary_labyrinth_pattern",
            "disordered_skyrmion_lattice_pattern",
        }
        if not enabled or pattern_type not in roi_pattern_types or pixel_size <= 0:
            return full_shape, None, None

        types = aperture_config.get("aperture_types", [])
        radii = aperture_config.get("aperture_radii", [])
        centers = aperture_config.get("aperture_centers", [])
        ellipticities = aperture_config.get("aperture_ellipticities", [1.0] * len(radii))
        sigmas = aperture_config.get("aperture_sigmas", [0.0] * len(radii))
        roughnesses = aperture_config.get("aperture_roughnesses", [0.0] * len(radii))
        top_radius_factors = aperture_config.get(
            "aperture_top_radius_factors", [2.0] * len(radii)
        )

        boxes: list[tuple[int, int, int, int]] = []
        center_y0 = full_shape[0] / 2
        center_x0 = full_shape[1] / 2

        for (
            aperture_type,
            radius,
            center,
            ellipticity,
            sigma,
            roughness,
            top_radius_factor,
        ) in zip(
            types,
            radii,
            centers,
            ellipticities,
            sigmas,
            roughnesses,
            top_radius_factors,
        ):
            if aperture_type != "OH":
                continue
            radius_px = float(radius) * max(1.0, float(top_radius_factor)) / pixel_size
            sigma_px = 0.0 if sigma is None else float(sigma) / pixel_size
            ellipticity = float(ellipticity)
            radius_y = radius_px * np.sqrt(ellipticity)
            radius_x = radius_px / np.sqrt(ellipticity)
            roughness_scale = 1.0 + max(0.0, 2.0 * float(roughness))
            half_y = radius_y * roughness_scale + 4.0 * sigma_px
            half_x = radius_x * roughness_scale + 4.0 * sigma_px
            margin = max(50.0, 0.25 * max(half_y, half_x))
            center_y = center_y0 + float(center[0]) / pixel_size
            center_x = center_x0 + float(center[1]) / pixel_size
            y0 = max(0, int(np.floor(center_y - half_y - margin)))
            y1 = min(full_shape[0], int(np.ceil(center_y + half_y + margin)))
            x0 = max(0, int(np.floor(center_x - half_x - margin)))
            x1 = min(full_shape[1], int(np.ceil(center_x + half_x + margin)))
            boxes.append((y0, y1, x0, x1))

        if not boxes:
            return full_shape, None, None

        y0 = min(box[0] for box in boxes)
        y1 = max(box[1] for box in boxes)
        x0 = min(box[2] for box in boxes)
        x1 = max(box[3] for box in boxes)
        roi_shape = (y1 - y0, x1 - x0)
        if roi_shape[0] <= 0 or roi_shape[1] <= 0:
            return full_shape, None, None

        slices = (slice(y0, y1), slice(x0, x1))
        coordinate_offset = (
            (y0 + y1) / 2 - full_shape[0] / 2,
            (x0 + x1) / 2 - full_shape[1] / 2,
        )
        return roi_shape, slices, coordinate_offset

    # ------------------------------------------------------------------
    # Core simulation — mirrors test.ipynb cell by cell
    # ------------------------------------------------------------------

    def _simulate_one(self, h5: h5py.File, idx: int) -> None:
        """Run one configuration and write holograms and metadata to HDF5.

        Parameters
        ----------
        h5 : h5py.File
            Open output HDF5 file.
        idx : int
            Zero-based sample index used for the output group name.

        Returns
        -------
        None
            The sample group is written into ``h5``.
        """
        stage_times: list[tuple[str, float]] = []

        def mark_stage(name: str, start: float) -> float:
            """Record elapsed time for a named pipeline stage.

            Parameters
            ----------
            name : str
                Human-readable stage name.
            start : float
                Start time from ``time.time()``.

            Returns
            -------
            now : float
                Current ``time.time()`` value, used as the next stage start.
            """
            now = time.time()
            stage_times.append((name, now - start))
            return now

        t_stage = time.time()
        p = self._sample_params()
        cfg = self.config
        metadata: dict[str, Any] = {}
        t_stage = mark_stage("sample params", t_stage)

        # ---- X-ray config ------------------------------------------------
        xray_config = XRayConfig(
            energy=p["energy"],
            pol="CR",
            photon_flux=p["photon_flux"],
            coherence_length=p["coherence_length"],
        )
        xray_config.setup()
        t_stage = mark_stage("xray", t_stage)

        # ---- Detector + beamstop config ----------------------------------
        detector_center = p["detector_center"] or tuple(
            np.array(cfg.detector_shape) // 2
        )
        detector_center = tuple(float(v) for v in detector_center)
        beamstop_values = dict(p["beamstop_config"])
        beamstop_center = tuple(detector_center)
        for center_key in ("beamstop_center", "bs_center", "center"):
            if center_key in beamstop_values:
                beamstop_center = tuple(float(v) for v in beamstop_values.pop(center_key))
                break
        for offset_key in ("beamstop_center_offset_px", "center_offset_px"):
            if offset_key in beamstop_values:
                offset = beamstop_values.pop(offset_key)
                beamstop_center = tuple(
                    float(c) + float(d) for c, d in zip(beamstop_center, offset)
                )
                break
        beamstop_config = BeamstopConfig(
            bs_method=cfg.beamstop_method,
            bs_detector_distance=cfg.beamstop_distance,
            bs_center=beamstop_center,
            bs_config=beamstop_values,
        )
        detector_config = DetectorConfig(
            pixel_size=p["detector_pixel_size"],
            shape=cfg.detector_shape,
            sample_to_detector_distance=p["detector_distance"],
            detector_center=detector_center,
            detector_params=p["detector_params"],
            artifacts_config=p["artifacts_config"],
            measurement_config=p["measurement_config"],
            use_detector_pixel_footprint=p["use_detector_pixel_footprint"],
            detector_pixel_footprint_samples=p["detector_pixel_footprint_samples"],
            ignore_flat_detector_curvature=p["ignore_flat_detector_curvature"],
            beamstop_config=beamstop_config,
        )
        detector_config.setup()
        real_space_pixel_size = (
            detector_config.calc_realspace_resolution(xray_config.beam_params)
            / cfg.oversampling
        )
        metadata.update(beamstop_config.get_metadata(prefix="beamstop/"))
        metadata.update(self._detector_metadata(detector_config))
        t_stage = mark_stage("detector/beamstop", t_stage)

        # ---- Sample config -----------------------------------------------
        sample_shape = np.array(
            [
                0,
                cfg.oversampling * cfg.detector_shape[0],
                cfg.oversampling * cfg.detector_shape[1],
            ],
            dtype=int,
        )
        sample_config = SampleConfig(
            recipe=cfg.recipe,
            sample_shape=sample_shape,
            real_space_pixel_size=real_space_pixel_size,
            xray_config=xray_config,
            sample_name=cfg.sample_name,
            sample_tilt_theta=cfg.sample_tilt_theta,
            sample_tilt_axis=cfg.sample_tilt_axis,
            sample_tilt_voxel_size=cfg.sample_tilt_voxel_size,
            sample_tilt_antialias_samples=cfg.sample_tilt_antialias_samples,
        )
        sample_config.setup()
        t_stage = mark_stage("sample setup", t_stage)
        sample_layer_names = sample_config.sample_structure.layer_names
        sample_layer_thicknesses = sample_config.sample_structure.layer_thicknesses
        membrane_index = next(
            (
                idx
                for idx, layer_name in enumerate(sample_layer_names)
                if FrontApertureConfig._is_silicon_nitride_layer(layer_name)
            ),
            len(sample_layer_names),
        )
        aperture_taper_depth = float(
            np.sum(sample_layer_thicknesses[: max(0, membrane_index - 2)])
        )
        thickness_oh = float(np.sum(sample_layer_thicknesses[:membrane_index]))

        # ---- Front aperture config ---------------------------------------
        front_aperture_config = FrontApertureConfig(
            aperture_method=cfg.aperture_method,
            aperture_shape=sample_shape,
            real_space_pixel_size=sample_config.sample_structure.real_space_pixel_size,
            aperture_thicknesses=sample_layer_thicknesses,
            aperture_layer_names=sample_layer_names,
            use_roi=cfg.use_roi,
            aperture_config=dict(
                apertures_type=p["aperture_config"]["aperture_types"],
                apertures_radius=p["aperture_config"]["aperture_radii"],
                apertures_depth=p["aperture_config"].get("aperture_depths"),
                apertures_length=p["aperture_config"].get("aperture_lengths"),
                apertures_center=p["aperture_config"]["aperture_centers"],
                apertures_sigma=p["aperture_config"]["aperture_sigmas"],
                apertures_angle=p["aperture_config"]["aperture_angles"],
                apertures_ellipticity=p["aperture_config"][
                    "aperture_ellipticities"
                ],
                apertures_roughness=p["aperture_config"]["aperture_roughnesses"],
                apertures_roughness_modes=p["aperture_config"][
                    "aperture_roughness_modes"
                ],
                apertures_seed=p["aperture_config"]["aperture_seeds"],
                apertures_top_radius_factor=p["aperture_config"][
                    "aperture_top_radius_factors"
                ],
                aperture_taper_depth=aperture_taper_depth,
                thickness_OH=thickness_oh,
            ),
        )

        # ---- Magnetic pattern config -------------------------------------
        pattern_shape, pattern_slices, coordinate_offset = self._magnetic_pattern_roi(
            sample_shape[1:],
            real_space_pixel_size,
            p["aperture_config"],
            p["pattern_type"],
            enabled=cfg.use_roi and cfg.magnetic_pattern_use_roi,
        )
        _, output_pattern_slices, _ = self._magnetic_pattern_roi(
            sample_shape[1:],
            real_space_pixel_size,
            p["aperture_config"],
            p["pattern_type"],
            enabled=cfg.use_roi,
        )
        pattern_config = dict(p["pattern_config"])
        if coordinate_offset is not None:
            pattern_config["coordinate_offset"] = coordinate_offset
        if p["pattern_type"] == "disordered_skyrmion_lattice_pattern":
            aperture_types = p["aperture_config"].get("aperture_types", [])
            if "OH" in aperture_types:
                oh_index = aperture_types.index("OH")
                oh_center = p["aperture_config"]["aperture_centers"][oh_index]
                oh_radius = p["aperture_config"]["aperture_radii"][oh_index]
                top_radius_factors = p["aperture_config"].get(
                    "aperture_top_radius_factors",
                    [2.0] * len(p["aperture_config"]["aperture_radii"]),
                )
                oh_top_radius = float(oh_radius) * max(
                    1.0, float(top_radius_factors[oh_index])
                )
                full_center_y = sample_shape[1] / 2 + float(oh_center[0]) / real_space_pixel_size
                full_center_x = sample_shape[2] / 2 + float(oh_center[1]) / real_space_pixel_size
                if pattern_slices is None:
                    placement_center = (full_center_y, full_center_x)
                else:
                    placement_center = (
                        full_center_y - pattern_slices[0].start,
                        full_center_x - pattern_slices[1].start,
                    )
                pattern_config.setdefault("placement_center", placement_center)
                pattern_config.setdefault(
                    "placement_radius",
                    oh_top_radius + float(pattern_config.get("stripe_width", 0.0)),
                )
        magnetic_pattern_config = MagneticPatternConfig(
            pattern_type_method=p["pattern_type"],
            shape=pattern_shape,
            real_space_pixel_size=real_space_pixel_size,
            pattern_config_length=p["pattern_config_length"],
            pattern_config=pattern_config,
        )
        magnetic_pattern_config.create_pattern()
        magnetic_pattern = np.ones(sample_shape[1:], dtype=np.float64)
        if pattern_slices is None:
            magnetic_pattern = magnetic_pattern_config.magnetic_pattern
        else:
            magnetic_pattern[pattern_slices] = magnetic_pattern_config.magnetic_pattern
        magnetization = pattern_generator.map_magnetization_to_3d(
            np.zeros_like(magnetic_pattern),
            np.sqrt(1 - np.abs(magnetic_pattern) ** 2),
            magnetic_pattern,
            nr_repeats=sample_shape[0],
        )
        sample_config.assign_magnetic_pattern(magnetization)
        magnetic_metadata = magnetic_pattern_config.get_metadata(
            prefix="sample/magnetic_pattern/"
        )
        magnetic_metadata.pop(
            "sample/magnetic_pattern/real_space_pixel_size", None
        )
        metadata.update(magnetic_metadata)
        if pattern_slices is not None:
            metadata["sample/magnetic_pattern/roi_y_start_px"] = (
                pattern_slices[0].start
            )
            metadata["sample/magnetic_pattern/roi_y_stop_px"] = (
                pattern_slices[0].stop
            )
            metadata["sample/magnetic_pattern/roi_x_start_px"] = (
                pattern_slices[1].start
            )
            metadata["sample/magnetic_pattern/roi_x_stop_px"] = (
                pattern_slices[1].stop
            )
        metadata["sample/use_roi"] = bool(cfg.use_roi)
        metadata["sample/tilt_theta_rad"] = float(cfg.sample_tilt_theta)
        metadata["sample/tilt_axis"] = str(cfg.sample_tilt_axis)
        if cfg.sample_tilt_voxel_size is not None:
            metadata["sample/tilt_voxel_size_m"] = float(cfg.sample_tilt_voxel_size)
        metadata["sample/tilt_antialias_samples"] = int(
            cfg.sample_tilt_antialias_samples
        )
        metadata["sample/magnetic_pattern/use_roi"] = bool(
            cfg.use_roi and cfg.magnetic_pattern_use_roi
        )
        t_stage = mark_stage("magnetic pattern", t_stage)

        front_aperture_config.setup()
        aperture_mask = front_aperture_config.return_aperture()
        sample_config.assign_aperture_mask(aperture_mask)
        aperture_material_fraction = np.mean(aperture_mask, axis=0)
        aperture_types = p["aperture_config"].get("aperture_types", [])
        aperture_centers = p["aperture_config"].get("aperture_centers", [])
        aperture_cut_pixels = []
        for center in aperture_centers:
            cut_y = int(
                np.clip(
                    round(sample_shape[1] / 2 + float(center[0]) / real_space_pixel_size),
                    0,
                    sample_shape[1] - 1,
                )
            )
            cut_x = int(
                np.clip(
                    round(sample_shape[2] / 2 + float(center[1]) / real_space_pixel_size),
                    0,
                    sample_shape[2] - 1,
                )
            )
            aperture_cut_pixels.append((cut_y, cut_x))
        if False:
            if aperture_cut_pixels:
                aperture_yz_cuts = np.stack(
                    [aperture_mask[:, :, cut_x] for _, cut_x in aperture_cut_pixels]
                )
                aperture_xz_cuts = np.stack(
                    [aperture_mask[:, cut_y, :] for cut_y, _ in aperture_cut_pixels]
                )
                metadata["aperture/cut_y_px_all"] = np.asarray(
                    [cut_y for cut_y, _ in aperture_cut_pixels], dtype=np.int64
                )
                metadata["aperture/cut_x_px_all"] = np.asarray(
                    [cut_x for _, cut_x in aperture_cut_pixels], dtype=np.int64
                )
            else:
                aperture_yz_cuts = np.empty((0, sample_shape[0], sample_shape[1]))
                aperture_xz_cuts = np.empty((0, sample_shape[0], sample_shape[2]))
            if "OH" in aperture_types:
                oh_index = aperture_types.index("OH")
                oh_center = p["aperture_config"]["aperture_centers"][oh_index]
            else:
                oh_center = (0.0, 0.0)
            aperture_cut_y = int(
                np.clip(
                    round(sample_shape[1] / 2 + float(oh_center[0]) / real_space_pixel_size),
                    0,
                    sample_shape[1] - 1,
                )
            )
            aperture_cut_x = int(
                np.clip(
                    round(sample_shape[2] / 2 + float(oh_center[1]) / real_space_pixel_size),
                    0,
                    sample_shape[2] - 1,
                )
            )
            aperture_yz_cut = aperture_mask[:, :, aperture_cut_x]
            aperture_xz_cut = aperture_mask[:, aperture_cut_y, :]
            metadata["aperture/cut_y_px"] = aperture_cut_y
            metadata["aperture/cut_x_px"] = aperture_cut_x

        supportmask = front_aperture_config.create_supportmask(
            output_shape=detector_config.detector_layout.detector_shape,
            output_pixel_size=detector_config.detector_layout.real_space_resolution,
        )
        oh_mask = front_aperture_config.create_supportmask(
            output_shape=sample_shape[1:],
            output_pixel_size=sample_config.sample_structure.real_space_pixel_size,
            aperture_types=("OH",),
        )
        magnetic_pattern_oh = magnetic_pattern * oh_mask
        domain_analysis = classify_magnetic_domains(
            magnetic_pattern,
            analysis_mask=np.asarray(oh_mask) > 0.5,
            min_area=cfg.magnetic_pattern_classification_min_area,
            max_eccentricity=cfg.magnetic_pattern_bubble_max_eccentricity,
            min_circularity=cfg.magnetic_pattern_bubble_min_circularity,
        )
        state = {
            "uniform": "saturated",
            "noise": "saturated",
            "bubbles": "bubble",
            "stripes": "stripe",
            "mixed": "mixed",
        }[domain_analysis["morphology"]]
        metadata["sample/magnetic_pattern/state"] = state
        metadata["sample/magnetic_pattern/bubble_count"] = domain_analysis[
            "bubble_count"
        ]
        metadata["sample/magnetic_pattern/stripe_count"] = domain_analysis[
            "stripe_count"
        ]
        metadata["sample/magnetic_pattern/noise_component_count"] = domain_analysis[
            "noise_count"
        ]
        metadata["sample/magnetic_pattern/classification_min_area_px"] = int(
            cfg.magnetic_pattern_classification_min_area
        )
        metadata["sample/magnetic_pattern/bubble_max_eccentricity"] = float(
            cfg.magnetic_pattern_bubble_max_eccentricity
        )
        metadata["sample/magnetic_pattern/bubble_min_circularity"] = float(
            cfg.magnetic_pattern_bubble_min_circularity
        )
        if output_pattern_slices is not None:
            magnetic_pattern_oh = magnetic_pattern_oh[output_pattern_slices]
            metadata["sample/magnetic_pattern/saved_roi_y_start_px"] = (
                output_pattern_slices[0].start
            )
            metadata["sample/magnetic_pattern/saved_roi_y_stop_px"] = (
                output_pattern_slices[0].stop
            )
            metadata["sample/magnetic_pattern/saved_roi_x_start_px"] = (
                output_pattern_slices[1].start
            )
            metadata["sample/magnetic_pattern/saved_roi_x_stop_px"] = (
                output_pattern_slices[1].stop
            )
        aperture_metadata = front_aperture_config.get_metadata(
            prefix="sample/aperture/"
        )
        aperture_metadata.pop("sample/aperture/real_space_pixel_size", None)
        aperture_metadata.pop("sample/aperture/use_roi", None)
        metadata.update(aperture_metadata)
        t_stage = mark_stage("front aperture", t_stage)

        if cfg.propagator_method == "Scalar":
            sample_config.sample_structure.calculate_final_scalar_refractive_index(
                pol="CR",
                use_aperture_roi=cfg.use_roi and cfg.dielectric_tensor_use_roi,
                compact=True,
                lazy=cfg.scalar_refractive_index_lazy,
            )
            metadata["sample/dielectric_tensor/skipped_for_scalar"] = True
            metadata["sample/scalar_refractive_index/use_roi"] = bool(
                cfg.use_roi and cfg.dielectric_tensor_use_roi
            )
            metadata["sample/scalar_refractive_index/compact"] = True
            metadata["sample/scalar_refractive_index/lazy"] = bool(
                cfg.scalar_refractive_index_lazy
            )
            t_stage = mark_stage("scalar refractive index setup", t_stage)
        else:
            beam_direction = _beam_direction_for_projected_jones(
                p["illumination_alpha_beam"]
            )
            sample_config.sample_structure.calculate_final_dielectric_tensor(
                use_aperture_roi=cfg.use_roi and cfg.dielectric_tensor_use_roi,
                compact=(
                    cfg.dielectric_tensor_compact
                    if beam_direction is None
                    and not p["dielectric_tensor_local_k_projection"]
                    else False
                ),
                beam_direction=beam_direction,
                local_k_projection=p["dielectric_tensor_local_k_projection"],
            )
            metadata["sample/dielectric_tensor/use_roi"] = bool(
                (cfg.use_roi and cfg.dielectric_tensor_use_roi)
                if beam_direction is None and not p["dielectric_tensor_local_k_projection"]
                else False
            )
            metadata["sample/dielectric_tensor/compact"] = bool(
                cfg.dielectric_tensor_compact
                if beam_direction is None and not p["dielectric_tensor_local_k_projection"]
                else False
            )
            metadata["sample/dielectric_tensor/beam_direction_projected"] = (
                beam_direction is not None
            )
            metadata["sample/dielectric_tensor/local_k_projected"] = bool(
                p["dielectric_tensor_local_k_projection"]
            )
            if beam_direction is not None:
                metadata["sample/dielectric_tensor/beam_direction_xyz"] = beam_direction
            t_stage = mark_stage("dielectric tensor", t_stage)

        # ---- Illumination config -----------------------------------------
        illumination_config = IlluminationConfig(
            XRayConfig=xray_config,
            shape=sample_shape[1:],
            real_space_pixel_size=real_space_pixel_size,
            illumination_function=cfg.illumination_function,
            illumination_config={
                "center": np.array(p["illumination_center"]),
                "distance": p["illumination_focus_distance"],
                "fwhm": p["illumination_fwhm"],
                "alpha_beam": p["illumination_alpha_beam"],
            },
        )
        illumination_config.setup()
        metadata["illumination/center_m"] = np.asarray(
            p["illumination_center"], dtype=float
        )
        metadata["illumination/function"] = str(cfg.illumination_function)
        metadata["illumination/focus_distance_m"] = p["illumination_focus_distance"]
        metadata["illumination/fwhm_m"] = p["illumination_fwhm"]
        metadata["illumination/alpha_beam_rad"] = p["illumination_alpha_beam"]
        t_stage = mark_stage("illumination", t_stage)

        # ---- Hologram simulation loop ------------------------------------
        hologram_config = HologramConfig(
            sample_x=sample_config.sample_structure.x,
            sample_y=sample_config.sample_structure.y,
            detector_layout=detector_config.detector_layout,
        )

        base_noise_seed = detector_config.detector_params.get("noise_seed")
        for pol_idx, pol in enumerate(self._POLARIZATIONS):
            if base_noise_seed is not None:
                detector_config.detector_params["noise_seed"] = (
                    int(base_noise_seed) + pol_idx
                )
            illumination_config.update_polarization(pol)

            propagator_config = SamplePropagatorConfig(
                SampleConfig=sample_config,
                IlluminationConfig=illumination_config,
                propagator_method=cfg.propagator_method,
                propagator_config={
                    "propagator_method": cfg.propagator_method,
                    "propagate": cfg.propagate,
                    "jones_apply_zero_order_phase": cfg.jones_apply_zero_order_phase,
                    "scalar_apply_zero_order_phase": cfg.jones_apply_zero_order_phase,
                    "scalar_refractive_index_lazy": cfg.scalar_refractive_index_lazy,
                    "dielectric_tensor_use_roi": (
                        cfg.use_roi and cfg.dielectric_tensor_use_roi
                    ),
                    "propagation_padding_px": cfg.propagation_padding_px,
                    "propagation_padding_mode": cfg.propagation_padding_mode,
                    "propagation_absorber_width_px": cfg.propagation_absorber_width_px,
                    "propagation_absorber_strength": cfg.propagation_absorber_strength,
                    "propagation_absorber_profile": cfg.propagation_absorber_profile,
                    "multislice_propagation_roi": cfg.multislice_propagation_roi,
                    "multislice_propagation_roi_padding_px": (
                        cfg.multislice_propagation_roi_padding_px
                    ),
                    "multislice_propagation_roi_merge_overlaps": (
                        cfg.multislice_propagation_roi_merge_overlaps
                    ),
                    "farfield_oversampling": cfg.farfield_oversampling,
                },
            )
            propagator_config.setup()
            t_stage = mark_stage(
                f"{pol} {cfg.propagator_method} propagation", t_stage
            )

            detector_config.assign_propagated_wavefront(propagator_config)
            detector_config.detect_hologram()
            t_stage = mark_stage(f"{pol} detector ideal", t_stage)

            hologram_config.add_exit_waves(
                {pol: propagator_config.return_scalar_wavefield()}
            )
            hologram_config.add_holograms(
                {pol: detector_config.return_ideal_hologram()}, source="ideal"
            )
            hologram_config.add_holograms(
                {
                    pol: detector_config.return_detected_hologram(
                        store_no_beamstop=(
                            cfg.save_detected_hologram_without_beamstop
                        )
                    )
                },
                source="detected",
            )
            artifact_model = detector_config.hologram_exp
            defect_map = artifact_model.camera_defect_map
            if pol_idx == 0:
                for defect_name in ("hot", "cold", "flicker"):
                    y_coords, x_coords = defect_map[defect_name]
                    metadata[
                        f"artifacts_realized/camera/{defect_name}_pixels_yx"
                    ] = np.column_stack((y_coords, x_coords)).astype(np.int64)
                    metadata[
                        f"artifacts_realized/camera/{defect_name}_baseline_values"
                    ] = defect_map[f"{defect_name}_baseline_values"]
                metadata["artifacts_realized/camera/hot_pixel_count"] = int(
                    defect_map["n_hot"]
                )
                metadata["artifacts_realized/camera/cold_pixel_count"] = int(
                    defect_map["n_cold"]
                )
            metadata[
                f"artifacts_realized/{pol}/cosmic_ray_events_yx_length_aspect_angle_value"
            ] = artifact_model.cosmic_ray_tracks
            metadata[f"artifacts_realized/{pol}/cosmic_ray_count"] = int(
                artifact_model.cosmic_ray_count
            )
            metadata[f"artifacts_realized/{pol}/effective_exposure_time_s"] = (
                1.0
                if artifact_model.exposure_time is None
                else float(artifact_model.exposure_time)
            )
            if cfg.save_detected_hologram_without_beamstop:
                hologram_config.add_holograms(
                    {
                        pol: detector_config.return_detected_hologram_without_beamstop()
                    },
                    source="detected_no_beamstop",
                )
            t_stage = mark_stage(f"{pol} detector noise", t_stage)

        metadata.update(sample_config.get_metadata(prefix="sample/"))
        metadata.update(self._propagator_metadata(propagator_config))
        metadata.update(self._xray_metadata(xray_config))
        metadata["detector/save_detected_no_beamstop"] = bool(
            cfg.save_detected_hologram_without_beamstop
        )

        # ---- Write to HDF5 ----------------------------------------------
        self._write_sample(
            h5,
            idx,
            hologram_config,
            detector_config,
            metadata,
            aperture_config=p["aperture_config"],
            supportmask=supportmask,
            magnetic_pattern_oh=magnetic_pattern_oh,
            #aperture_material_fraction=aperture_material_fraction,
            #aperture_yz_cut=aperture_yz_cut,
            #aperture_xz_cut=aperture_xz_cut,
            #aperture_yz_cuts=aperture_yz_cuts,
            #aperture_xz_cuts=aperture_xz_cuts,
        )
        mark_stage("hdf5 write", t_stage)

        if self.verbose:
            breakdown = ", ".join(
                f"{name}: {elapsed:.2f}s" for name, elapsed in stage_times
            )
            print(f"    timing: {breakdown}")

    # ------------------------------------------------------------------
    # HDF5 I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _detector_metadata(detector_config: DetectorConfig) -> dict[str, Any]:
        """Return detector metadata with configuration groups kept as siblings."""
        raw = detector_config.get_metadata(prefix="detector/")
        result: dict[str, Any] = {}
        sibling_groups = (
            "detector_params",
            "measurement_config",
            "artifacts_config",
        )
        for key, value in raw.items():
            remapped = key
            for group in sibling_groups:
                prefix = f"detector/{group}/"
                if key.startswith(prefix):
                    remapped = f"{group}/{key[len(prefix):]}"
                    break
            result[remapped] = value
        return result

    @staticmethod
    def _xray_metadata(xray_config: XRayConfig) -> dict[str, Any]:
        """Return one canonical X-ray metadata group."""
        beam = xray_config.beam_params
        return {
            "xray/energy_eV": beam.energy,
            "xray/photon_flux": beam.photon_flux,
            "xray/coherence_length_m": beam.coherence_length,
            "xray/wavelength_m": beam.wavelength,
            "xray/wavevector_per_m": beam.wavevector,
        }

    def _write_sample(
        self,
        h5: h5py.File,
        idx: int,
        hologram_config: HologramConfig,
        detector_config: DetectorConfig,
        metadata: dict[str, Any],
        aperture_config: dict[str, Any],
        supportmask: np.ndarray,
        magnetic_pattern_oh: np.ndarray,
        #aperture_material_fraction: np.ndarray,
        #aperture_yz_cut: np.ndarray,
        #aperture_xz_cut: np.ndarray,
        #aperture_yz_cuts: np.ndarray,
        #aperture_xz_cuts: np.ndarray,
    ) -> None:
        """Write one simulation's holograms and metadata to an HDF5 group.

        Parameters
        ----------
        h5 : h5py.File
            Open output HDF5 file.
        idx : int
            Zero-based sample index used for the output group name.
        hologram_config : HologramConfig
            Container with exit waves, ideal holograms, and detected holograms.
        detector_config : DetectorConfig
            Detector configuration containing the detector layout and beamstop.
        metadata : dict[str, Any]
            Scalar metadata to write under the sample ``metadata`` group.
        aperture_config : dict[str, Any]
            Aperture configuration to store with typed HDF5 datasets.
        supportmask : np.ndarray
            Detector-space support mask to save.
        magnetic_pattern_oh : np.ndarray
            Object-hole magnetic pattern crop to save.

        Returns
        -------
        None
            Datasets are written into the sample group.
        """
        grp = h5.create_group(f"{idx:05d}", track_order=True)

        # Hologram arrays — same structure as test.ipynb save_data dict
        save_arrays = hologram_config.to_dict(
            helicities=["CR", "CL"],
            sources=[
                "exit_wave",
                "ideal",
                "detected",
                "detected_no_beamstop",
            ],
        )
        for helicity, per_source in save_arrays.items():
            for source, arr in per_source.items():
                arr = np.asarray(arr)
                arr = arr.astype(
                    np.complex64 if np.iscomplexobj(arr) else np.float32,
                    copy=False,
                )
                grp.create_dataset(
                    f"{helicity}/{source}", data=arr, compression="gzip"
                )

        # Beamstop mask
        if hasattr(detector_config.detector_layout, "beamstop"):
            grp.create_dataset(
                "beamstop_mask",
                data=np.asarray(detector_config.detector_layout.beamstop),
                compression="gzip",
            )

        grp.create_dataset(
            "supportmask",
            data=np.asarray(supportmask, dtype=np.uint8),
            compression="gzip",
        )
        grp.create_dataset(
            "magnetic_pattern_oh",
            data=np.asarray(magnetic_pattern_oh, dtype=np.float32),
            compression="gzip",
        )
        if False:
            grp.create_dataset(
                "aperture_material_fraction",
                data=np.asarray(aperture_material_fraction, dtype=np.float32),
                compression="gzip",
            )
            grp.create_dataset(
                "aperture_yz_cut",
                data=np.asarray(aperture_yz_cut, dtype=np.float32),
                compression="gzip",
            )
            grp.create_dataset(
                "aperture_xz_cut",
                data=np.asarray(aperture_xz_cut, dtype=np.float32),
                compression="gzip",
            )
            grp.create_dataset(
                "aperture_yz_cuts",
                data=np.asarray(aperture_yz_cuts, dtype=np.float32),
                compression="gzip",
            )
            grp.create_dataset(
                "aperture_xz_cuts",
                data=np.asarray(aperture_xz_cuts, dtype=np.float32),
                compression="gzip",
            )

        self._write_metadata(grp, metadata, aperture_config)

    def _write_metadata(
        self,
        sample_grp: h5py.Group,
        metadata: dict[str, Any],
        aperture_config: dict[str, Any],
    ) -> None:
        """Write the canonical, ordered metadata hierarchy for one sample."""
        meta_grp = sample_grp.create_group("metadata", track_order=True)
        self._write_aperture_metadata(meta_grp, aperture_config)

        metadata_items = list(metadata.items())
        prop_method_items = [
            item
            for item in metadata_items
            if item[0] == "propagator_config/propagator_method"
        ]
        prop_config_items = [
            item
            for item in metadata_items
            if item[0].startswith("propagator_config/")
            and item[0] != "propagator_config/propagator_method"
        ]
        other_items = [
            item
            for item in metadata_items
            if not item[0].startswith("propagator_config/")
        ]

        for key, value in prop_method_items + prop_config_items + other_items:
            if key.startswith("sample/aperture/aperture_config/apertures_"):
                continue
            parts = key.split("/")
            sub = meta_grp
            for part in parts[:-1]:
                if part in sub:
                    sub = sub[part]
                else:
                    sub = sub.create_group(part, track_order=True)
            if isinstance(value, str):
                value = np.bytes_(value)
            try:
                sub.create_dataset(parts[-1], data=value)
            except (TypeError, ValueError):
                pass  # skip non-serialisable values (arrays, complex objects)

    def _write_aperture_metadata(
        self, meta_grp: h5py.Group, aperture_config: dict[str, Any]
    ) -> None:
        """Store typed aperture lists under metadata/sample/aperture.

        Parameters
        ----------
        meta_grp : h5py.Group
            Sample metadata group.
        aperture_config : dict[str, Any]
            Aperture configuration containing type, geometry, roughness, and
            seed lists.

        Returns
        -------
        None
            Aperture datasets are written into ``meta_grp``.
        """
        sample_grp = (
            meta_grp["sample"]
            if "sample" in meta_grp
            else meta_grp.create_group("sample", track_order=True)
        )
        aperture_grp = sample_grp.create_group("aperture", track_order=True)
        aperture_grp = aperture_grp.create_group("aperture_config", track_order=True)
        aperture_grp.create_dataset(
            "apertures_type",
            data=np.asarray(aperture_config["aperture_types"], dtype="S"),
        )
        aperture_grp.create_dataset(
            "apertures_radius",
            data=np.asarray(aperture_config["aperture_radii"], dtype=np.float64),
        )
        depth_values = aperture_config.get("aperture_depths")
        if depth_values is None:
            depth_values = [np.nan] * len(aperture_config["aperture_types"])
        aperture_grp.create_dataset(
            "apertures_depth",
            data=np.asarray(
                [np.nan if value is None else value for value in depth_values],
                dtype=np.float64,
            ),
        )
        aperture_grp.create_dataset(
            "apertures_length",
            data=np.asarray(
                aperture_config.get(
                    "aperture_lengths",
                    [0.0] * len(aperture_config["aperture_types"]),
                ),
                dtype=np.float64,
            ),
        )
        aperture_grp.create_dataset(
            "apertures_center",
            data=np.asarray(aperture_config["aperture_centers"], dtype=np.float64),
        )
        aperture_grp.create_dataset(
            "apertures_sigma",
            data=np.asarray(aperture_config["aperture_sigmas"], dtype=np.float64),
        )
        aperture_grp.create_dataset(
            "apertures_angle",
            data=np.asarray(aperture_config["aperture_angles"], dtype=np.float64),
        )
        aperture_grp.create_dataset(
            "apertures_ellipticity",
            data=np.asarray(
                aperture_config["aperture_ellipticities"], dtype=np.float64
            ),
        )
        aperture_grp.create_dataset(
            "apertures_roughness",
            data=np.asarray(aperture_config["aperture_roughnesses"], dtype=np.float64),
        )
        aperture_grp.create_dataset(
            "apertures_roughness_modes",
            data=np.asarray(
                aperture_config["aperture_roughness_modes"], dtype=np.int64
            ),
        )
        aperture_grp.create_dataset(
            "apertures_seed",
            data=np.asarray(aperture_config["aperture_seeds"], dtype=np.int64),
        )
        aperture_grp.create_dataset(
            "apertures_top_radius_factor",
            data=np.asarray(
                aperture_config.get(
                    "aperture_top_radius_factors",
                    [2.0] * len(aperture_config["aperture_types"]),
                ),
                dtype=np.float64,
            ),
        )

    @staticmethod
    def _fth_reconstruct(hologram: np.ndarray) -> np.ndarray:
        """Compute an FTH reconstruction over the last two hologram axes.

        Parameters
        ----------
        hologram : np.ndarray
            Hologram array. The Fourier transform is applied over the last two
            axes, so stacked leading dimensions are preserved.

        Returns
        -------
        reconstruction : np.ndarray
            Complex FTH reconstruction with the same shape as ``hologram``.
        """
        return np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(hologram, axes=(-2, -1)), axes=(-2, -1)),
            axes=(-2, -1),
        )

    def _write_pipeline_config(self, h5: h5py.File) -> None:
        """Store file-global pipeline parameters as datasets.

        Parameters
        ----------
        h5 : h5py.File
            Open output HDF5 file.

        Returns
        -------
        None
            The ``_pipeline_config`` group is written into ``h5``.
        """
        cfg = self.config
        grp = h5.create_group("_pipeline_config")
        grp.create_dataset("n_samples", data=self.n_samples)
        grp.create_dataset("oversampling", data=cfg.oversampling)
