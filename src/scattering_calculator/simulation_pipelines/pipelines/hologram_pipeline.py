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

from scattering_calculator.sample_generator import pattern_generator
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


@dataclass
class HologramPipelineConfig:
    """Fixed physical parameters shared across all simulation runs.

    Any parameter can be overridden or swept using :class:`HologramPipelineRanges`.
    Physical lengths are in SI units (metres, eV).

    Parameters
    ----------
    recipe : str
        Multilayer thin-film recipe, e.g. ``"Au(700)/Cr(300)/SiN(200)/Co(90)/Pt(120)/Al(60)"``.
        Thicknesses are in angstroms, ordered top-to-bottom.
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
        RMS readout noise in detector counts.
    detector_quantum_efficiency : float
        Detector quantum efficiency (0–1).
    beamstop_method : {"circular"} or None
        Beamstop shape. ``None`` → transparent (no beamstop).
    beamstop_distance : float
        Sample-to-beamstop distance in metres.
    beamstop_config : dict
        Optional keyword arguments forwarded to the beamstop generator. If
        ``"radius"`` is missing, no beamstop or wire is created. Beamstop
        lengths, including ``"sigma"``, are in metres.
    aperture_method : {"FTH_circular"} or None
        Holography mask layout. ``None`` → fully transparent.
    aperture_types : list of {"OH", "RH"}
        Object hole (OH) or reference hole (RH) for each aperture.
    aperture_radii : list of float
        Aperture radii in metres.
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
    pattern_type : {"wavy_stripe_pattern", "skyrmion_pattern"}
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
    oversampling : int
        Oversampling factor relative to the Nyquist limit from the detector.
        ``real_space_pixel_size = detector_resolution / oversampling``.
        Default 2.
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
    detector_noise_rms: float = 3.0
    detector_quantum_efficiency: float = 1.0
    detector_params: dict = field(
        default_factory=lambda: {
            "readout_noise_average": 50,
            "noise_rms": 3,
            "detector_threshold": 64e3,
            "counts_per_photon": 100,
            "quantum_efficiency": 1.0,
        }
    )
    artifacts_config: dict = field(
        default_factory=lambda: {
            "counts_per_photon": 100,
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

    # Beamstop
    beamstop_method: str | None = "circular"
    beamstop_distance: float = 0.001  # m
    beamstop_config: dict = field(default_factory=dict)

    # FTH holography mask
    aperture_method: str | None = "FTH_circular"
    aperture_types: list[str] = field(
        default_factory=lambda: ["OH", "RH", "RH"]
    )
    aperture_radii: list[float] = field(
        default_factory=lambda: [60e-9, 6e-9, 4e-9]
    )  # m
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

    # Illumination
    illumination_function: str | None = "gaussian"
    illumination_center: tuple[float, float] = (0.0, 0.0)  # m
    illumination_focus_distance: float = 1e-3  # m
    illumination_fwhm: float = 0.5e-6  # m

    # Magnetic domain pattern
    pattern_type: str = "wavy_stripe_pattern"
    pattern_config: dict = field(default_factory=dict)
    pattern_config_length: dict = field(default_factory=dict)

    # Simulation grid
    oversampling: int = 2


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
        tuple[float, float] | tuple[Uniform, Uniform] | Uniform | None
    ) = None

    # Detector
    detector_distance: float | Uniform | Callable[[dict[str, Any]], Any] | None = None
    detector_pixel_size: float | Uniform | None = None
    detector_noise_rms: float | Uniform | None = None
    detector_params: dict | None = None
    artifacts_config: dict | None = None
    measurement_config: dict | None = None

    # Beamstop
    beamstop_config: dict | None = None

    # FTH holography mask
    aperture_config: dict | Callable[[dict[str, Any]], dict] | None = None

    # Magnetic pattern
    pattern_type: str | Choice | None = None
    pattern_config: dict | Callable[[dict[str, Any]], dict] | None = None
    pattern_config_length: dict | None = None


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
        ├── _pipeline_config/       ← top-level fixed parameters
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
        │       ├── magnetic_pattern/
        │       └── aperture/
        │           └── aperture_config/
        │               ├── apertures_type    ← aperture labels, e.g. OH/RH
        │               ├── apertures_radius  ← aperture radii in metres
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
        self.config = config
        self.ranges = ranges
        self.output_path = Path(output_path)
        self.n_samples = n_samples
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the simulation loop and write all results to HDF5."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        width = len(str(self.n_samples))
        t_start = time.time()
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
        if self.verbose:
            elapsed = time.time() - t_start
            print(
                f"Pipeline complete: {self.n_samples} configurations in "
                f"{elapsed:.1f} s → {self.output_path}"
            )

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def _sample_params(self) -> dict[str, Any]:
        """Draw one sampled parameter set, merging ranges over the base config."""
        cfg = self.config
        rng = self.ranges

        def _resolve(value: Any, params: dict[str, Any]) -> Any:
            if callable(value):
                value = value(params)
            if isinstance(value, tuple):
                return tuple(_resolve(v, params) for v in value)
            return _s(value)

        def _pick(range_val: Any, base_val: Any, params: dict[str, Any]) -> Any:
            return base_val if range_val is None else _resolve(range_val, params)

        def _sample_dict_with_params(d: dict, params: dict[str, Any]) -> dict:
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
            merged = dict(base_dict)
            if range_dict is not None:
                if callable(range_dict):
                    range_dict = _resolve(range_dict, params)
                merged.update(_sample_dict_with_params(range_dict, params))
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
        params["detector_pixel_size"] = _pick(
            rng.detector_pixel_size, cfg.detector_pixel_size, params
        )
        params["detector_noise_rms"] = _pick(
            rng.detector_noise_rms, cfg.detector_noise_rms, params
        )
        params["detector_params"] = _merge_dict(
            rng.detector_params, cfg.detector_params, params
        )
        if rng.detector_noise_rms is not None:
            params["detector_params"]["noise_rms"] = params["detector_noise_rms"]
        elif (
            "noise_rms" not in params["detector_params"]
            and "readout_noise_sigma" not in params["detector_params"]
        ):
            params["detector_params"]["noise_rms"] = params["detector_noise_rms"]
        if (
            cfg.detector_quantum_efficiency != 1.0
            and (
                rng.detector_params is None
                or "quantum_efficiency" not in rng.detector_params
            )
        ):
            params["detector_params"]["quantum_efficiency"] = (
                cfg.detector_quantum_efficiency
            )
        elif rng.detector_params is None and "quantum_efficiency" not in cfg.detector_params:
            params["detector_params"]["quantum_efficiency"] = (
                cfg.detector_quantum_efficiency
            )
        params["detector_params"].setdefault("quantum_efficiency", 1.0)
        params["artifacts_config"] = _merge_dict(
            rng.artifacts_config, cfg.artifacts_config, params
        )
        params["measurement_config"] = _merge_dict(
            rng.measurement_config, cfg.measurement_config, params
        )
        params["pattern_type"] = _pick(rng.pattern_type, cfg.pattern_type, params)
        params["pattern_config"] = _merge_dict(
            rng.pattern_config, cfg.pattern_config, params
        )
        params["pattern_config_length"] = _merge_dict(
            rng.pattern_config_length, cfg.pattern_config_length, params
        )
        for key, value in params["pattern_config_length"].items():
            params["pattern_config"].setdefault(key, value)
        params["detector_distance"] = _pick(
            rng.detector_distance, cfg.detector_distance, params
        )
        params["beamstop_config"] = _merge_dict(
            rng.beamstop_config, cfg.beamstop_config, params
        )
        params["aperture_config"] = _merge_dict(
            rng.aperture_config,
            {
                "aperture_types": cfg.aperture_types,
                "aperture_radii": cfg.aperture_radii,
                "aperture_centers": cfg.aperture_centers,
                "aperture_sigmas": cfg.aperture_sigmas,
                "aperture_angles": cfg.aperture_angles,
                "aperture_ellipticities": cfg.aperture_ellipticities,
                "aperture_roughnesses": cfg.aperture_roughnesses,
                "aperture_roughness_modes": cfg.aperture_roughness_modes,
                "aperture_seeds": cfg.aperture_seeds,
            },
            params,
        )
        return params

    # ------------------------------------------------------------------
    # Core simulation — mirrors test.ipynb cell by cell
    # ------------------------------------------------------------------

    def _simulate_one(self, h5: h5py.File, idx: int) -> None:
        """Run one configuration and write holograms + metadata to HDF5."""
        stage_times: list[tuple[str, float]] = []

        def mark_stage(name: str, start: float) -> float:
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
        detector_center = cfg.detector_center or tuple(
            np.array(cfg.detector_shape) // 2
        )
        beamstop_config = BeamstopConfig(
            bs_method=cfg.beamstop_method,
            bs_detector_distance=cfg.beamstop_distance,
            bs_center=tuple(detector_center),
            bs_config=p["beamstop_config"],
        )
        detector_config = DetectorConfig(
            pixel_size=p["detector_pixel_size"],
            shape=cfg.detector_shape,
            sample_to_detector_distance=p["detector_distance"],
            detector_center=detector_center,
            detector_params=p["detector_params"],
            artifacts_config=p["artifacts_config"],
            measurement_config=p["measurement_config"],
            beamstop_config=beamstop_config,
        )
        detector_config.setup()
        real_space_pixel_size = (
            detector_config.calc_realspace_resolution(xray_config.beam_params)
            / cfg.oversampling
        )
        metadata.update(beamstop_config.get_metadata(prefix="beamstop/"))
        metadata.update(detector_config.get_metadata(prefix="detector/"))
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
        )
        sample_config.setup()
        t_stage = mark_stage("sample setup", t_stage)

        # ---- Magnetic pattern config -------------------------------------
        magnetic_pattern_config = MagneticPatternConfig(
            pattern_type_method=p["pattern_type"],
            shape=sample_shape[1:],
            real_space_pixel_size=real_space_pixel_size,
            pattern_config_length=p["pattern_config_length"],
            pattern_config=p["pattern_config"],
        )
        magnetic_pattern_config.create_pattern()
        magnetic_pattern = magnetic_pattern_config.magnetic_pattern
        magnetization = pattern_generator.map_magnetization_to_3d(
            np.zeros_like(magnetic_pattern),
            np.sqrt(1 - np.abs(magnetic_pattern) ** 2),
            magnetic_pattern,
            nr_repeats=sample_shape[0],
        )
        sample_config.assign_magnetic_pattern(magnetization)
        metadata.update(magnetic_pattern_config.get_metadata(prefix="magnetic_pattern/"))
        t_stage = mark_stage("magnetic pattern", t_stage)

        # ---- Front aperture config ---------------------------------------
        front_aperture_config = FrontApertureConfig(
            aperture_method=cfg.aperture_method,
            aperture_shape=sample_shape,
            real_space_pixel_size=sample_config.sample_structure.real_space_pixel_size,
            aperture_thicknesses=sample_config.sample_structure.layer_thicknesses,
            aperture_config=dict(
                apertures_type=p["aperture_config"]["aperture_types"],
                apertures_radius=p["aperture_config"]["aperture_radii"],
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
                thickness_OH=float(
                    np.sum(
                        sample_config.sample_structure.layer_thicknesses[
                            : sample_config.sample_structure.layer_names.index("SiN")
                        ]
                    )
                ),
            ),
        )
        front_aperture_config.setup()
        sample_config.assign_aperture_mask(front_aperture_config.return_aperture())
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
        metadata.update(front_aperture_config.get_metadata(prefix="aperture/"))
        t_stage = mark_stage("front aperture", t_stage)

        sample_config.sample_structure.calculate_final_dielectric_tensor()
        t_stage = mark_stage("dielectric tensor", t_stage)

        # ---- Illumination config -----------------------------------------
        illumination_config = IlluminationConfig(
            XRayConfig=xray_config,
            shape=sample_shape[1:],
            real_space_pixel_size=real_space_pixel_size,
            illumination_function=cfg.illumination_function,
            illumination_config={
                "center": np.array(cfg.illumination_center),
                "distance": cfg.illumination_focus_distance,
                "fwhm": cfg.illumination_fwhm,
            },
        )
        illumination_config.setup()
        t_stage = mark_stage("illumination", t_stage)

        # ---- Hologram simulation loop ------------------------------------
        hologram_config = HologramConfig(
            sample_x=sample_config.sample_structure.x,
            sample_y=sample_config.sample_structure.y,
            detector_layout=detector_config.detector_layout,
        )

        for pol in self._POLARIZATIONS:
            illumination_config.update_polarization(pol)

            propagator_config = SamplePropagatorConfig(
                SampleConfig=sample_config,
                IlluminationConfig=illumination_config,
                propagator_method="Jones",
            )
            propagator_config.setup()
            t_stage = mark_stage(f"{pol} Jones propagation", t_stage)

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
                {pol: detector_config.return_detected_hologram()}, source="detected"
            )
            t_stage = mark_stage(f"{pol} detector noise", t_stage)

        metadata.update(propagator_config.get_metadata())
        metadata.update(xray_config.get_metadata(prefix="xray/"))

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
    ) -> None:
        """Write one simulation's holograms and metadata to an HDF5 group."""
        grp = h5.create_group(f"{idx:05d}", track_order=True)

        # Hologram arrays — same structure as test.ipynb save_data dict
        save_arrays = hologram_config.to_dict(
            helicities=["CR", "CL"],
            sources=["exit_wave", "ideal", "detected"],
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

        # Metadata — scalar datasets under metadata/ subgroup
        meta_grp = grp.create_group("metadata")
        self._write_aperture_metadata(meta_grp, aperture_config)
        for key, value in metadata.items():
            if key.startswith("aperture/aperture_config/apertures_"):
                continue
            parts = key.split("/")
            sub = meta_grp
            for part in parts[:-1]:
                sub = sub.require_group(part)
            if isinstance(value, str):
                value = np.bytes_(value)
            try:
                sub.create_dataset(parts[-1], data=value)
            except (TypeError, ValueError):
                pass  # skip non-serialisable values (arrays, complex objects)

    def _write_aperture_metadata(
        self, meta_grp: h5py.Group, aperture_config: dict[str, Any]
    ) -> None:
        """Store aperture lists as typed HDF5 datasets under metadata/aperture."""
        aperture_grp = meta_grp.require_group("aperture").require_group(
            "aperture_config"
        )
        aperture_grp.create_dataset(
            "apertures_type",
            data=np.asarray(aperture_config["aperture_types"], dtype="S"),
        )
        aperture_grp.create_dataset(
            "apertures_radius",
            data=np.asarray(aperture_config["aperture_radii"], dtype=np.float64),
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

    @staticmethod
    def _fth_reconstruct(hologram: np.ndarray) -> np.ndarray:
        """FTH reconstruction over the last two axes of a hologram array."""
        return np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(hologram, axes=(-2, -1)), axes=(-2, -1)),
            axes=(-2, -1),
        )

    def _write_pipeline_config(self, h5: h5py.File) -> None:
        """Store top-level fixed pipeline parameters as datasets."""
        cfg = self.config
        grp = h5.create_group("_pipeline_config")
        grp.create_dataset("recipe", data=np.bytes_(cfg.recipe))
        grp.create_dataset("n_samples", data=self.n_samples)
        grp.create_dataset("oversampling", data=cfg.oversampling)
        grp.create_dataset("aperture_method", data=np.bytes_(str(cfg.aperture_method)))
        grp.create_dataset(
            "illumination_function", data=np.bytes_(str(cfg.illumination_function))
        )
        grp.create_dataset("beamstop_method", data=np.bytes_(str(cfg.beamstop_method)))
        grp.create_dataset("beamstop_distance_m", data=cfg.beamstop_distance)
        beamstop_cfg_grp = grp.create_group("beamstop_config")
        for key, value in cfg.beamstop_config.items():
            if isinstance(value, str):
                value = np.bytes_(value)
            try:
                beamstop_cfg_grp.create_dataset(key, data=value)
            except (TypeError, ValueError):
                pass
        grp.create_dataset("illumination_fwhm_m", data=cfg.illumination_fwhm)
        grp.create_dataset(
            "illumination_focus_distance_m", data=cfg.illumination_focus_distance
        )
        grp.create_dataset("detector_shape", data=np.array(cfg.detector_shape))
        grp.create_dataset(
            "detector_quantum_efficiency", data=cfg.detector_quantum_efficiency
        )
        self._write_config_dict(grp, "detector_params", cfg.detector_params)
        self._write_config_dict(grp, "measurement_config", cfg.measurement_config)
        self._write_config_dict(grp, "artifacts_config", cfg.artifacts_config)

    def _write_config_dict(self, parent: h5py.Group, name: str, config: dict) -> None:
        """Write scalar/list config values to a subgroup."""
        grp = parent.require_group(name)
        for key, value in config.items():
            if isinstance(value, str):
                value = np.bytes_(value)
            try:
                grp.create_dataset(key, data=value)
            except (TypeError, ValueError):
                pass
