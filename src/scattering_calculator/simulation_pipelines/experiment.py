"""Common experiment definition and execution for notebooks and scripts.

All physical lengths are metres, photon energies eV, and angles radians.
Recipe thickness literals remain nm, as in SampleConfig throughout the package.
The sections use the existing production configuration classes, not example APIs.
"""
from __future__ import annotations

from copy import deepcopy, copy
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from .simulation_configuration import (
    XRayConfig, SimulationConfig, SampleConfig, MagneticPatternConfig,
    FrontApertureConfig, IlluminationConfig, SamplePropagatorConfig,
    DetectorConfig, BeamstopConfig,
)
from scattering_calculator.experimental_conditions import light_beam


@dataclass
class OutputConfig:
    """HDF5 destination and saved observables for the common experiment runner.

    Paths are relative to the working directory unless absolute. ``save`` names
    data products; ``plots`` selects panels for run_experiment's summary figure.
    Metadata and physical coordinates are always saved. Large sample volumes
    are opt-in. Existing files require explicit ``overwrite=True``.
    """
    path: str | Path = "outputs/experiment.h5"
    save: tuple[str, ...] = ("exit_wave", "fft_intensity", "detector_ideal", "detector_measured")
    compression: str | None = "gzip"
    overwrite: bool = False
    plots: tuple[str, ...] = ("exit_wave", "fft_intensity", "detector_ideal", "detector_measured")
    figure_path: str | Path | None = None


def analyze_jones(field, angle):
    """Project coherent Jones channels onto an ideal laboratory linear analyzer.

    Axis 2 is polarization; an optional axis 3 stores incoherent Stokes carriers.
    Returns amplitudes. Square magnitudes and sum incoherent carriers afterward.

    Parameters
    ----------
    field : ndarray
        Coherent Jones channels on the sample or padded FFT grid.
    angle : float
        Laboratory analyzer angle in radians from +x towards +y.

    Returns
    -------
    ndarray
        Projected complex amplitudes, retaining any incoherent mode axis.
    """
    field = np.asarray(field)
    if field.ndim not in (3, 4) or field.shape[2] != 2:
        raise ValueError("An analyzer requires coherent Jones fields, not scalar intensities")
    if not np.isfinite(angle):
        raise ValueError("Analyzer angle must be finite")
    return np.cos(angle)*field[:,:,0] + np.sin(angle)*field[:,:,1]


@dataclass
class ExperimentConfig:
    """The same ordered setup sections for every scattering experiment.

    Component references/grids are wired by ScatteringExperiment.setup().
    ``analysis`` contains numerical settings for postprocessing, never specimen,
    illumination, propagation or camera parameters.
    """
    xray: XRayConfig = field(default_factory=lambda: XRayConfig(778., 1.))
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    sample: SampleConfig = field(default_factory=lambda: SampleConfig(recipe="Co(30)", sample_name="Co film"))
    magnetic_pattern: MagneticPatternConfig = field(default_factory=lambda: MagneticPatternConfig(pattern_type_method="saturated_pattern"))
    aperture: FrontApertureConfig = field(default_factory=lambda: FrontApertureConfig(aperture_method=None))
    illumination: IlluminationConfig = field(default_factory=lambda: IlluminationConfig(illumination_function=None))
    propagation: SamplePropagatorConfig = field(default_factory=lambda: SamplePropagatorConfig(propagator_config={"propagate": True}))
    detector: DetectorConfig = field(default_factory=lambda: DetectorConfig(detector_center=(128, 128)))
    beamstop: BeamstopConfig = field(default_factory=lambda: BeamstopConfig(bs_method=None))
    energies_eV: tuple[float, ...] | None = None  # compatibility alias; scans live in illumination
    outputs: OutputConfig = field(default_factory=OutputConfig)
    analysis: dict = field(default_factory=dict)

    def scan_axes(self):
        """Resolve single-exposure defaults and explicit illumination scan axes.

        Parameters
        ----------
        None
            Uses source defaults and illumination scan lists.

        Returns
        -------
        tuple
            Energy and polarization tuples in execution order (energy then state).
        """
        energies = self.illumination.energies_eV
        if energies is not None and self.energies_eV is not None and not np.array_equal(energies, self.energies_eV):
            raise ValueError("Conflicting scan energies: use illumination.energies_eV")
        if energies is None:
            energies = self.energies_eV if self.energies_eV is not None else (self.xray.energy,)
        energies = tuple(float(value) for value in energies)
        if not energies or not np.isfinite(energies).all() or min(energies) <= 0:
            raise ValueError("Scan energies must be a nonempty list of finite positive eV values")
        states = self.illumination.polarizations
        if isinstance(states, str):
            raise ValueError("polarizations must be a list or tuple, e.g. ('CR', 'CL')")
        states = tuple(states) if states is not None else (self.xray.pol,)
        if not states or any(state not in ('CR','CL','LH','LV','x','y') for state in states):
            raise ValueError("Scan polarizations must use CR, CL, LH, LV, x or y")
        return energies, states

    def to_dict(self):
        """Return the standard experiment fields for provenance.

        Parameters
        ----------
        None
            Uses the stored experiment state.

        Returns
        -------
        dict
            Dataclass fields for provenance of the experiment declaration.
        """
        result = asdict(self)
        result['outputs']['path'] = str(self.outputs.path)
        if self.outputs.figure_path is not None:
            result['outputs']['figure_path'] = str(self.outputs.figure_path)
        return result

    def with_changes(self, **sections):
        """Return an independent config, e.g. with_changes(xray={"energy": 780.}).

        Parameters
        ----------
        sections : dict
            Section replacements; mappings update fields of existing component classes.

        Returns
        -------
        ExperimentConfig
            Independent declaration with the requested component updates.
        """
        result = deepcopy(self)
        for name, values in sections.items():
            if name not in self.__dataclass_fields__:
                raise TypeError(f"Unknown experiment section: {name}")
            current = getattr(result, name)
            setattr(result, name, replace(current, **values) if isinstance(values, dict) and hasattr(current, "__dataclass_fields__") else deepcopy(values))
        return result

    def to_pipeline_config(self):
        """Translate the standard definition for the existing HDF5 batch backend.

        Kept at the backend boundary; notebooks declare only ExperimentConfig.

        Parameters
        ----------
        None
            Uses the stored experiment state.

        Returns
        -------
        HologramPipelineConfig
            Backend record after validating supported experiment settings.
        """
        from .pipelines.hologram_pipeline import HologramPipelineConfig
        x,s,m,a,i,p,d,b = (self.xray,self.sample,self.magnetic_pattern,self.aperture,
                          self.illumination,self.propagation,self.detector,self.beamstop)
        profile, holes = i.illumination_config, a.aperture_config
        values = dict(recipe=s.recipe,sample_name=s.sample_name,
            xray_energy=x.energy,xray_photon_flux=x.photon_flux,xray_coherence_length=x.coherence_length,
            detector_shape=d.shape,detector_pixel_size=d.pixel_size,
            detector_distance=d.sample_to_detector_distance,detector_center=d.detector_center,
            detector_params=d.detector_params,artifacts_config=d.artifacts_config,measurement_config=d.measurement_config,
            detector_propagation_method=d.detector_propagation_method,
            use_detector_pixel_footprint=d.use_detector_pixel_footprint,
            detector_pixel_footprint_samples=d.detector_pixel_footprint_samples,
            ignore_flat_detector_curvature=d.ignore_flat_detector_curvature,
            beamstop_method=b.bs_method,beamstop_distance=b.bs_detector_distance,beamstop_config=b.bs_config,
            save_detected_hologram_without_beamstop=d.save_detected_hologram_without_beamstop,
            aperture_method=a.aperture_method,
            use_roi=a.use_roi,
            illumination_function=i.illumination_function,
            illumination_center=profile.get('center',(0.,0.)),
            illumination_focus_distance=profile.get('distance',0.),illumination_fwhm=profile.get('fwhm',.5e-6),
            illumination_alpha_beam=profile.get('alpha_beam',(0.,0.)),
            pattern_type=m.pattern_type_method,pattern_config=m.pattern_config,pattern_config_length=m.pattern_config_length,
            propagator_method=p.propagator_method,
            sample_tilt_theta=s.sample_tilt_theta,sample_tilt_axis=s.sample_tilt_axis,
            sample_tilt_voxel_size=s.sample_tilt_voxel_size,sample_tilt_antialias_samples=s.sample_tilt_antialias_samples,
            oversampling=self.simulation.other_config.get('oversampling',2),
            random_seed=self.simulation.other_config.get('random_seed'),
        )
        aperture_names = {'types':'type','radii':'radius','lengths':'length','depths':'depth',
            'centers':'center','sigmas':'sigma','angles':'angle','ellipticities':'ellipticity',
            'roughnesses':'roughness','roughness_modes':'roughness_modes','seeds':'seed','top_radius_factors':'top_radius_factor'}
        for key,suffix in aperture_names.items():
            if 'apertures_'+suffix in holes:
                values['aperture_'+key] = holes['apertures_'+suffix]
        if 'apertures_type' in holes:
            count = len(holes['apertures_type'])
            defaults = {'lengths':0.,'sigmas':0.,'angles':0.,'ellipticities':1.,
                'roughnesses':0.,'roughness_modes':(0,0),'seeds':-1,'top_radius_factors':2.}
            for key,default in defaults.items():
                values.setdefault('aperture_'+key,[default]*count)
            values['aperture_seeds'] = [-1 if seed is None else seed for seed in values['aperture_seeds']]
            if 'thickness_OH' in holes and 'aperture_depths' not in values:
                values['aperture_depths'] = [holes['thickness_OH'] if kind == 'OH' else None
                                             for kind in holes['apertures_type']]
        known = HologramPipelineConfig.__dataclass_fields__
        values.update({k:v for k,v in p.propagator_config.items() if k in known})
        if x.linear_polarization_angle is not None or d.analyzer_angle is not None or s.max_slice_thickness is not None:
            raise ValueError('The legacy paired-CR/CL HDF5 backend cannot represent linear angles, analyzers or slice subdivision; use ScatteringExperiment for these experiments')
        if self.simulation.other_config.get('grid_mode','explicit') != 'detector':
            raise ValueError('The HDF5 sweep backend currently requires simulation.other_config grid_mode="detector"; use ScatteringExperiment for an explicit grid')
        return HologramPipelineConfig(**values)


def vector_pattern(config):
    """Vector examples registered on the standard MagneticPatternConfig.

    Parameters
    ----------
    config : MagneticPatternConfig
        Pattern settings bound to a source grid.

    Returns
    -------
    tuple
        Unit vector map and its (x,y) coordinates in metres.
    """
    if config.shape is None or config.real_space_pixel_size is None:
        raise ValueError("Bind magnetic_pattern to a simulation grid before creating it")
    ny, nx = config.shape
    yy, xx = np.indices((ny, nx), dtype=float)
    x = (xx-nx//2)*config.real_space_pixel_size
    y = (yy-ny//2)*config.real_space_pixel_size
    p = config.resolved_pattern_config()
    if config.pattern_type_method == "magnon_wave":
        period = p.get("period", 40e-9)
        amplitude = p.get("amplitude", .1)
        angle = p.get("angle", 0.)
        mean_angle = p.get("inplane_angle", 0.)
        if not np.isfinite(period) or period <= 0 or not np.isfinite(amplitude) or not 0 <= amplitude < 1:
            raise ValueError("Magnon period must be positive and 0 <= amplitude < 1")
        if not np.isfinite([angle, mean_angle, p.get("phase", 0.)]).all():
            raise ValueError("Magnetization angles and phase must be finite")
        mz = amplitude*np.sin(2*np.pi*(x*np.cos(angle)+y*np.sin(angle))/period+p.get("phase", 0.))
        ip = np.sqrt(1-mz*mz)
        m = np.stack((ip*np.cos(mean_angle), ip*np.sin(mean_angle), mz), axis=-1)
    elif config.pattern_type_method == "smooth_domains":
        period = p.get("period", 110e-9)
        if not np.isfinite(period) or period <= 0:
            raise ValueError("Domain period must be positive")
        mz = np.tanh((np.sin(2*np.pi*x/period+1.4*np.sin(y/90e-9))+.5*np.cos(2*np.pi*y/160e-9))/.22)
        m = np.stack((np.sqrt(1-mz*mz), np.zeros_like(mz), mz), axis=-1)
    else:
        lattice, radius = p.get("period", 12e-9), p.get("radius", 3.4e-9)
        if not 0 < radius < lattice/2:
            raise ValueError("Neel texture requires 0 < radius < period/2")
        m = np.zeros((ny, nx, 3)); m[...,2] = 1.
        # Same compact radial texture convention as the independent paper reference.
        extent = max(abs(x).max(), abs(y).max())+lattice
        span = int(np.ceil(2*extent/lattice))+2
        for j in range(-span, span+1):
            cy = j*np.sqrt(3)*lattice/2
            for i in range(-span, span+1):
                cx = (i+.5*(j % 2))*lattice
                dx, dy = x-cx, y-cy
                r = np.hypot(dx, dy); inside = r < radius
                u = r[inside]/radius
                polar = np.pi*(1-3*u*u+2*u*u*u)
                azimuth = np.arctan2(dy[inside], dx[inside])
                m[inside,0] = np.sin(polar)*np.cos(azimuth)
                m[inside,1] = np.sin(polar)*np.sin(azimuth)
                m[inside,2] = np.cos(polar)
    return m, (x, y)


class ScatteringExperiment:
    """setup → propagate → detect, shared by all complete experiment examples."""
    def __init__(self, config: ExperimentConfig):
        """Store the standard experiment declaration.

        Parameters
        ----------
        config : ExperimentConfig
            Standard source, specimen and instrument declaration.

        Returns
        -------
        None
            Completes in place or raises if the asserted contract is violated.
        """
        if not isinstance(config, ExperimentConfig):
            raise TypeError("Use ExperimentConfig with the standard component sections")
        self.config = deepcopy(config)
        if config.illumination.spectral_components is not None:
            self._spectral_declaration = deepcopy(config)

    @classmethod
    def from_exit_wave(cls, config, exit_wave):
        """Continue the standard detector stage from a supplied coherent exit field.

        Used for independent propagation benchmarks, without inventing a sample
        or illumination model. Source energy and grid still use standard sections.
        Padding is zero exterior for this explicitly supplied finite field.

        Parameters
        ----------
        config : ExperimentConfig
            Standard source, specimen and instrument declaration.
        exit_wave : ndarray
            Supplied coherent scalar or Jones field on the configured source grid.

        Returns
        -------
        ScatteringExperiment
            Prepared run that starts at the supplied exit plane.
        """
        result = cls(config)
        c = result.config
        field = np.array(exit_wave,dtype=complex,copy=True)
        if field.shape[:2] != tuple(c.simulation.shape) or field.ndim not in (2,3):
            raise ValueError('Exit field must match simulation.shape and have scalar or Jones channels')
        if field.ndim == 3 and field.shape[-1] != 2:
            raise ValueError('Jones exit fields require two components')
        if not np.isfinite(field).all(): raise ValueError('Exit field must be finite')
        factor = c.propagation.propagator_config.get('farfield_oversampling',1)
        if int(factor) != factor or factor < 1: raise ValueError('farfield_oversampling must be a positive integer')
        pads = [(n*(int(factor)-1)//2,n*(int(factor)-1)-n*(int(factor)-1)//2) for n in field.shape[:2]]
        if field.ndim == 3: pads.append((0,0))
        extended = np.pad(field,pads)
        fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(extended,axes=(0,1)),axes=(0,1)),axes=(0,1))
        power = abs(fourier)**2
        result.wavefront = SimpleNamespace(exit_wave=field,exit_wave_for_farfield=extended,
            hologram=power if power.ndim == 2 else power.sum(axis=-1))
        c.propagation.propagator_method = 'Scalar' if field.ndim == 2 else 'Jones'
        c.propagation.wavefront = result.wavefront
        c.propagation.SampleConfig = c.sample
        c.propagation.IlluminationConfig = c.illumination
        c.sample.real_space_pixel_size = c.simulation.real_space_pixel_size
        c.illumination.beam_params = c.xray.setup()
        c.detector.beamstop_config = c.beamstop
        c.detector.setup()
        c.detector.calc_realspace_resolution(c.illumination.beam_params)
        return result

    def setup(self, *, magnetization=None, mask=None):
        """Build grid, optical data, specimen, apertures, illumination and camera.

        Optional mask and vector arrays support imported/custom specimens without
        defining another configuration format. Arrays must use the configured grid.

        Parameters
        ----------
        magnetization : ndarray or None
            Optional vector texture, (Ny,Nx,3) or (Nz,Ny,Nx,3).
        mask : ndarray or None
            Optional material occupancy on the subdivided specimen grid.

        Returns
        -------
        ScatteringExperiment
            This run with all source/grid references and component stages prepared.
        """
        c = self.config
        if c.simulation.other_config.get('grid_mode') == 'detector':
            factor = c.simulation.other_config.get('oversampling',2)
            if not isinstance(factor,int) or factor < 1:
                raise ValueError('Detector-derived grid oversampling must be a positive integer')
            c.xray.setup()
            c.detector.setup()
            c.simulation.shape = tuple(factor*n for n in c.detector.shape)
            c.simulation.real_space_pixel_size = c.detector.calc_realspace_resolution(c.xray.beam_params)/factor
        c.simulation.setup()
        shape, pitch = c.simulation.shape, c.simulation.real_space_pixel_size
        c.xray.setup()
        c.sample.sample_shape = (0, *shape)
        c.sample.real_space_pixel_size = pitch
        c.sample.xray_config = c.xray
        c.sample.setup()
        sample = c.sample.sample_structure
        nz = len(sample.layer_thicknesses)
        c.aperture.aperture_shape = (nz, *shape)
        c.aperture.real_space_pixel_size = pitch
        c.aperture.aperture_thicknesses = list(sample.layer_thicknesses)
        c.aperture.aperture_layer_names = list(sample.layer_names)
        if mask is None:
            if c.aperture.aperture_method is None:
                mask = np.ones((nz, *shape))  # no hole: retain the film
            else:
                c.aperture.setup()
                mask = c.aperture.return_aperture()
        mask = np.asarray(mask)
        if mask.shape != (nz, *shape):
            raise ValueError("Mask shape must match the subdivided sample grid")
        c.sample.assign_aperture_mask(mask)
        c.magnetic_pattern.shape = shape
        c.magnetic_pattern.real_space_pixel_size = pitch
        if magnetization is None:
            generated, _ = c.magnetic_pattern.create_pattern()
            if generated.shape == shape:
                generated = np.stack((np.zeros_like(generated),np.zeros_like(generated),generated),axis=-1)
            if generated.shape != (*shape, 3):
                raise ValueError("Magnetic pattern must be (Ny,Nx) or (Ny,Nx,3)")
            magnetization = np.zeros((nz, *shape, 3))
            for iz, name in enumerate(sample.layer_names):
                if any(material in name for material in c.magnetic_pattern.magnetic_materials):
                    magnetization[iz] = generated
        magnetization = np.asarray(magnetization)
        if magnetization.shape == (*shape, 3):
            magnetization = np.broadcast_to(magnetization, (nz,*shape,3))
        if magnetization.shape != (nz,*shape,3) or not np.isfinite(magnetization).all():
            raise ValueError("Magnetization must be finite and match the sample grid")
        c.sample.assign_magnetic_pattern(magnetization)
        c.illumination.XRayConfig = c.xray
        c.illumination.shape = shape
        c.illumination.real_space_pixel_size = pitch
        if c.illumination.illumination_function == "gaussian" and c.illumination.illumination_config.get("center") is None:
            c.illumination.illumination_config = {**c.illumination.illumination_config, "center":(0.,0.)}
        c.illumination.setup()
        c.propagation.SampleConfig = c.sample
        c.propagation.IlluminationConfig = c.illumination
        c.detector.beamstop_config = c.beamstop
        c.detector.setup()
        c.detector.calc_realspace_resolution(c.illumination.beam_params)
        self.sample, self.illumination = sample, c.illumination
        return self

    def propagate(self):
        """Store the standard experiment declaration.

        Parameters
        ----------
        None
            Uses the stored experiment state.

        Returns
        -------
        object
            Production wavefront with coherent exit channels and optional far field.
        """
        if not hasattr(self, "sample"):
            raise RuntimeError("Call setup() before propagate()")
        c = self.config
        if c.illumination.spectral_components is not None:
            specifications = c.illumination.spectral_components
            if not specifications:
                raise ValueError("spectral_components must contain at least one component")
            weights = np.asarray([item.weight for item in specifications], dtype=float)
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                raise ValueError("Spectral component weights must be finite and positive")
            total_weight = weights.sum()
            if not np.isfinite(total_weight):
                raise ValueError("Sum of spectral weights must be finite")
            weights /= total_weight
            self.spectral_runs = []
            self.spectral_weights = weights
            for item, fraction in zip(specifications, weights):
                energy = item.energy if item.energy is not None else c.xray.energy * item.energy_factor
                if not np.isfinite(energy) or energy <= 0:
                    raise ValueError("Spectral component energy must be finite and positive")
                component = self._spectral_declaration.with_changes(
                    xray={"energy":energy, "photon_flux":c.xray.photon_flux*float(fraction),
                          "pol":item.pol or c.xray.pol},
                    illumination={"spectral_components":None})
                run = ScatteringExperiment(component).setup(
                    magnetization=self.sample.magnetization, mask=self.sample.mask)
                run.propagate()
                self.spectral_runs.append(run)
            self.wavefront = self.spectral_runs[0].wavefront
            return self.wavefront
        if c.propagation.propagator_method != "Scalar":
            direction = light_beam.beam_direction_from_alpha(c.illumination.illumination_config.get("alpha_beam", (0.,0.)))
            c.sample.sample_structure.calculate_final_dielectric_tensor(
                compact=bool(c.propagation.propagator_config.get("dielectric_tensor_compact", True)),
                use_aperture_roi=bool(c.propagation.propagator_config.get("dielectric_tensor_use_roi", True)),
                beam_direction=None if np.allclose(direction,[0,0,1]) else direction,
                local_k_projection=bool(c.propagation.propagator_config.get("dielectric_tensor_local_k_projection", False)),
            )
        c.propagation.setup()
        self.wavefront = c.propagation.return_wavefront()
        return self.wavefront

    def detect(self, *, noise=False):
        """Detect the configured analyzer image.

        Parameters
        ----------
        noise : bool
            Apply acquisition and sensor response when True.

        Returns
        -------
        ndarray
            Ideal or measured detector intensity on the configured camera grid.
        """
        if not hasattr(self, "wavefront"):
            raise RuntimeError("Call propagate() before detect()")
        c = self.config
        if hasattr(self, "spectral_runs"):
            images = []
            for component in self.spectral_runs:
                component.config.detector = DetectorConfig(**{item.name:deepcopy(getattr(c.detector,item.name))
                    for item in fields(c.detector) if item.name != "beamstop_config"})
                component.config.detector.beamstop_config = deepcopy(c.beamstop)
                component.config.detector.setup()
                component.spectral_ideal = component.detect().copy()
                images.append(component.spectral_ideal)
            # Components carry their flux fractions in the field amplitudes.
            # Sum detector intensities, never complex amplitudes or noisy frames.
            c.detector = copy(self.spectral_runs[0].config.detector)
            c.detector.hologram_exp = copy(c.detector.hologram_exp)
            c.detector.hologram_exp.hologram_detector = np.sum(images, axis=0)
            return c.detector.return_detected_hologram() if noise else c.detector.return_ideal_hologram()
        if self.wavefront.hologram is None:
            # Exit-only Jones runs defer the FFT until the detector is requested.
            from scattering_calculator.utils.image_transformator import Fraunhofer_propagation_jones
            factor = self.wavefront.farfield_oversampling
            shape = tuple(n * factor for n in self.wavefront.exit_wave.shape[:2])
            background = c.propagation._background_exit_jones(shape) if factor > 1 else None
            field = self.wavefront._build_farfield_exit_wave(self.wavefront.exit_wave, background, factor)
            self.wavefront.exit_wave_for_farfield = field
            self.wavefront.detector_wave = Fraunhofer_propagation_jones(field)
            self.wavefront.hologram = np.sum(abs(self.wavefront.detector_wave)**2, axis=-1)
        c.detector.assign_propagated_wavefront(c.propagation)
        projection = c.illumination.beam_params
        if c.detector.projection_energy is not None:
            energy = c.detector.projection_energy
            if not np.isfinite(energy) or energy <= 0:
                raise ValueError("Detector projection_energy must be finite and positive")
            source = deepcopy(c.xray)
            source.energy = energy
            projection = source.setup()
        c.detector.calc_realspace_resolution(projection)
        c.detector.detect_hologram(projection_beam_params=projection)
        return c.detector.return_detected_hologram() if noise else c.detector.return_ideal_hologram()

    def run(self, *, noise=False, **setup_arrays):
        """Store the standard experiment declaration.

        Parameters
        ----------
        noise : bool
            Apply acquisition and sensor response when True.
        setup_arrays : dict
            Custom arrays passed to setup.

        Returns
        -------
        ndarray
            Detector image after setup, sample propagation and detection.
        """
        self.setup(**setup_arrays)
        self.propagate()
        return self.detect(noise=noise)
