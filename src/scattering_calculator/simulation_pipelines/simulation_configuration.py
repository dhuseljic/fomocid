from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from scattering_calculator.experimental_conditions import detector, light_beam
from scattering_calculator.sample_generator import pattern_generator
from scattering_calculator.sample_generator import structures
from scattering_calculator.beam_propagator import Jones_propagator


class _ConfigMixin:
    def to_dict(self) -> dict:
        """Return all configuration fields and their current values as a dict."""
        return dataclasses.asdict(self)


@dataclass
class XRayConfig(_ConfigMixin):
    """Configuration for the X-ray source.

    Parameters
    ----------
    energy : float
        Photon energy in eV.
    photon_flux : float
        Photon flux in photons per pulse.
    polarization : {"CR", "CL", "x", "y"}
        Polarization state of the beam.
    coherence_length : float
        Transverse coherence length in metres.

    Attributes
    ----------
    beam_params : light_beam.beam_parameters
        Populated by ``setup()``.
    wavevector : float
        Photon wavevector magnitude in 1/m, populated by ``setup()``.
    """

    energy: float  # eV
    photon_flux: float  # photons/s
    pol: Literal["CR", "CL", "x", "y"] = "CR"
    coherence_length: float = 10e-6  # m

    def __post_init__(self) -> None:
        if self.energy <= 0:
            raise ValueError(f"energy must be positive, got {self.energy}")
        if self.photon_flux <= 0:
            raise ValueError(f"photon_flux must be positive, got {self.photon_flux}")
        if self.coherence_length <= 0:
            raise ValueError(
                f"coherence_length must be positive, got {self.coherence_length}"
            )
        if self.pol not in ["CR", "CL", "x", "y"]:
            raise ValueError(
                f"Polarisation must be in CR, CL, x or, got {self.polarization}"
            )

    def setup(self) -> light_beam.beam_parameters:
        """Instantiate beam parameters and compute the wavevector.

        Returns
        -------
        light_beam.beam_parameters
            The configured beam parameter object.
        """
        self.beam_params = light_beam.beam_parameters(
            self.energy, self.pol, self.photon_flux, self.coherence_length
        )
        self.beam_params.calc_wavevector()
        return self.beam_params


@dataclass
class BeamstopConfig(_ConfigMixin):
    """Configuration for the beamstop placed between sample and detector.

    Parameters
    ----------
    bs_method : {"circular"} or None
        Shape of the beamstop. ``None`` creates an empty (transparent) beamstop.
    bs_detector_distance : float
        Distance from sample to beamstop plane in metres.
    bs_center : tuple of int
        Beamstop centre position in pixels (row, col).
    bs_config : dict
        Additional keyword arguments forwarded to the beamstop creation method
        (e.g. ``{"radius": 0.5e-3}`` for a circular beamstop).
    """

    bs_method: Literal["circular", None] | None = "circular"
    bs_detector_distance: float = 0.01  # m
    bs_center: tuple[int, int] = (0, 0)  # px
    bs_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bs_detector_distance <= 0:
            raise ValueError(
                f"bs_detector_distance must be positive, got {self.bs_detector_distance}"
            )

    def setup(self, detector_layout: detector.detector_layout) -> detector.beamstop:
        """Create the beamstop object and attach it to the detector layout.

        Parameters
        ----------
        detector_layout : detector.detector_layout
            The detector geometry to which the beamstop belongs.

        Returns
        -------
        detector.beamstop
            The configured beamstop object.
        """
        bs = detector.beamstop(
            detector_config=detector_layout,
            distance_detector_beamstop=self.bs_detector_distance,
        )
        if self.bs_method is None:
            return bs.create_empty_beamstop()
        if self.bs_method == "circular":
            bs.create_circle_beamstop(
                center=self.bs_center, use_real_space_coordinates=True, **self.bs_config
            )
        return bs


@dataclass
class DetectorConfig(_ConfigMixin):
    """Configuration for the detector geometry and noise properties.

    Parameters
    ----------
    shape : tuple of int
        Detector size in pixels (rows, cols).
    pixel_size : float
        Physical pixel size in metres.
    sample_to_detector_distance : float
        Sample-to-detector distance in metres.
    detector_center : tuple of int
        Position of the direct beam on the detector in pixels (row, col).
    artifacts_method : str or None
        Method for simulating detector artifacts. Not yet implemented.
    artifacts_config : dict
        Parameters for the artifact simulation method.
    beamstop : BeamstopConfig or None
        Optional beamstop configuration. If provided, the beamstop is created
        and assigned to the detector layout during ``setup()``.

    Attributes
    ----------
    detector_layout : detector.detector_layout
        Populated by ``setup()``.
    """

    shape: tuple[int, int] = (256, 256)  # px
    pixel_size: float = 55e-6  # m/px
    sample_to_detector_distance: float = 0.1  # m
    detector_center: tuple[int, int] = (0, 0)  # px
    detector_params: dict = field(
        default_factory=lambda: {"quantum_efficiency": 1.0, "noise_rms": 0.0}
    )
    artifacts_method: str | None = None
    artifacts_config: dict = field(default_factory=dict)
    measurement_config: dict = field(default_factory=lambda: {"number_frames": 1})
    beamstop_config: BeamstopConfig | None = None

    def __post_init__(self) -> None:
        if any(s <= 0 for s in self.shape):
            raise ValueError(f"shape dimensions must be positive, got {self.shape}")
        if self.pixel_size <= 0:
            raise ValueError(f"pixel_size must be positive, got {self.pixel_size}")
        if self.sample_to_detector_distance <= 0:
            raise ValueError(
                f"sample_to_detector_distance must be positive, got {self.sample_to_detector_distance}"
            )

    def setup(self) -> detector.detector_layout:
        """Build the detector layout and optionally attach the beamstop.

        Returns
        -------
        detector.detector_layout
            The configured detector layout object.
        """
        self.detector_layout = detector.detector_layout(
            pixel_size=self.pixel_size,
            detector_shape=self.shape,
            distance_sample_detector=self.sample_to_detector_distance,
            detector_center=self.detector_center,
        )
        if self.beamstop_config is not None:
            self.beamstop = self.beamstop_config.setup(self.detector_layout)
            self.detector_layout.assign_beamstop(self.beamstop)

        self.detector_layout.calc_real_space_coordinates()

    def calc_realspace_resolution(
        self, beam_parameters: light_beam.beam_parameters
    ) -> float:
        """Compute the real-space resolution limit imposed by the detector.

        Parameters
        ----------
        beam_parameters : light_beam.beam_parameters
            Beam parameters used to compute the q-space coordinates.

        Returns
        -------
        float
            Real-space resolution in metres.
        """
        self.detector_layout.calc_q_space_coordinates(beam_parameters)
        self.detector_layout.calc_resolution_from_detector()

        return self.detector_layout.real_space_resolution

    def assign_propagated_wavefront(self, samplepropagationconfig) -> None:
        """Assign a simulated hologram to the detector layout for later retrieval."""
        self.propagator = samplepropagationconfig
        self.wavefront = self.propagator.return_wavefront()

    def detect_hologram(self) -> np.ndarray:
        """Simulate the detection of the hologram on the detector, including noise and artifacts."""
        if not hasattr(self, "wavefront"):
            raise ValueError("No propagated wavefront assigned to detector layout.")
        self.hologram_exp = detector.detector_hologram(
            self.detector_layout,
            self.wavefront.hologram,
            self.propagator.IlluminationConfig.beam_params,
            self.propagator.SampleConfig.real_space_pixel_size,
            self.beamstop,
        )
        self.hologram_exp.gnomonic_projection()

    def return_ideal_hologram(self) -> np.ndarray:
        """Return the ideal (noise-free, artifact-free) hologram as a 2-D array."""
        if not hasattr(self, "wavefront"):
            raise ValueError("No propagated wavefront assigned to detector layout.")
        return self.hologram_exp.hologram_detector

    def return_detected_hologram(self) -> np.ndarray:
        """Return the simulated detected hologram as a 2-D array."""
        self.hologram_exp.add_noise()
        if not hasattr(self, "hologram_exp"):
            raise ValueError("No hologram detected yet. Call detect_hologram() first.")
        return self.hologram_exp.hologram_exp


@dataclass
class SimulationConfig(_ConfigMixin):
    """Configuration for the real-space simulation grid.

    Parameters
    ----------
    shape : tuple of int
        Number of pixels along (rows, cols).
    real_space_pixel_size : float
        Physical size of one pixel in metres.
    other_config : dict
        Reserved for future extension.

    Attributes
    ----------
    xgrid, ygrid : ndarray
        2-D real-space coordinate grids in metres, populated by ``setup()``.
    """

    shape: tuple[int, int] = (256, 256)  # px
    real_space_pixel_size: float = 10e-9  # m/px
    other_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(s <= 0 for s in self.shape):
            raise ValueError(f"shape dimensions must be positive, got {self.shape}")
        if self.real_space_pixel_size <= 0:
            raise ValueError(
                f"real_space_pixel_size must be positive, got {self.real_space_pixel_size}"
            )

    def setup(self) -> None:
        """Build centred real-space coordinate grids and store them as ``xgrid``/``ygrid``."""
        y = (np.arange(self.shape[0]) - self.shape[0] / 2) * self.real_space_pixel_size
        x = (np.arange(self.shape[1]) - self.shape[1] / 2) * self.real_space_pixel_size
        self.xgrid, self.ygrid = np.meshgrid(x, y)


@dataclass
class SampleConfig(_ConfigMixin):
    """Configuration for a multilayer thin-film sample structure.

    Parameters
    ----------
    recipe : str
        Layer stack recipe string, e.g. ``"Au(700)/Cr(300)/SiN(200)/Co(90)"``.
        Thicknesses are in angstroms.
    sample_shape : tuple of int
        Sample array shape ``(Nz, Ny, Nx)`` in pixels. ``Nz = 0`` is a
        sentinel that gets updated to the number of layers after ``setup()``.
    real_space_pixel_size : float
        Physical pixel size in metres.
    xray_config : XRayConfig
        X-ray source configuration used to look up refractive indices.
    sample_name : str or None
        Human-readable label for the sample.
    comments : str or None
        Free-text annotation stored in the multilayer recipe.

    Attributes
    ----------
    multilayer_recipe : structures.MultilayerRecipe
        Parsed layer stack, populated by ``setup()``.
    material_params : structures.material_params
        Refractive index database for all materials in the stack, populated by ``setup()``.
    sample_structure : structures.Structure
        The assembled layered structure object, populated by ``setup()``.
    """

    recipe: str = ("Recipe",)
    sample_shape: tuple[int, int, int] | None = (None,)
    real_space_pixel_size: float | None = (None,)
    xray_config: XRayConfig | None = (None,)
    sample_name: str | None = (None,)
    comments: str | None = (None,)
    other_config: dict = field(default_factory=dict)

    def setup(self) -> None:
        """Parse the recipe, load refractive indices, and build the layer stack."""
        self.multilayer_recipe = structures.parse_recipe(
            self.recipe,
            sample_name=self.sample_name,
            comments=self.comments,
        )

        self.material_params = structures.material_params(
            materials=set(
                [element.material for element in self.multilayer_recipe.layers]
            ),
            x_ray_energy=self.xray_config.energy,
        )

        self.sample_structure = structures.Structure(
            name=self.sample_name,
            material_params=self.material_params,
            sample_shape=self.sample_shape,
            real_space_pixel_size=self.real_space_pixel_size,
        )
        for layer in self.multilayer_recipe.layers:
            self.sample_structure.add_layer(layer.material, thickness=layer.thickness)

    def assign_magnetic_pattern(self, magnetic_vector_field: np.ndarray) -> None:
        """Map a 2-D scalar magnetization pattern onto the 3-D sample stack.

        Parameters
        ----------
        magnetic_vector_field : ndarray of shape (nr_layer,mx, my, mz)
            (4-D) 3-D magnetic vector field.
        """
        self.sample_structure.magnetization = magnetic_vector_field

    def assign_aperture_mask(self, aperture_mask: np.ndarray) -> None:
        """Attach a 3-D aperture mask and store its 2-D projection.

        Parameters
        ----------
        aperture_mask : ndarray of shape (Nz, Ny, Nx)
            3-D binary/soft aperture mask.
        """
        self.sample_structure.mask = aperture_mask
        self.sample_structure.aperture_mask2D = np.average(aperture_mask, axis=0)


@dataclass
class MagneticPatternConfig(_ConfigMixin):
    """Configuration for generating a 2-D magnetic domain pattern.

    Parameters
    ----------
    pattern_type_method : {"skyrmion_pattern", "wavy_stripe_pattern"}
        Which pattern generator function to use.
    shape : tuple of int or None
        2-D array shape ``(Ny, Nx)`` in pixels.
    real_space_pixel_size : float or None
        Physical pixel size in metres. Used to convert ``pattern_config_length``
        values to pixel units before calling the generator.
    pattern_config : dict
        Dimensionless or non-length parameters forwarded directly to the
        generator (e.g. ``number_skyr``, ``angle_stripes``, ``seed``).
    pattern_config_length : dict
        Physical-length parameters in metres (e.g. ``skyr_radius``,
        ``stripe_width``). Each value is divided by ``real_space_pixel_size``
        before being forwarded to the generator.

    Attributes
    ----------
    magnetic_pattern : ndarray of shape (Ny, Nx)
        Out-of-plane magnetization map, populated by ``create_pattern()``.
    pattern_coordinates : ndarray
        Auxiliary coordinate output from the generator, populated by
        ``create_pattern()``.
    """

    pattern_type_method: Literal["skyrmion_pattern", "wavy_stripe_pattern"] = (
        "skyrmion_pattern"
    )
    shape: tuple[int, int] | None = None
    real_space_pixel_size: float | None = None
    pattern_config: dict = field(default_factory=dict)
    pattern_config_length: dict = field(default_factory=dict)

    def setup(self):
        """Return the generator function selected by ``pattern_type_method``."""
        _methods = {
            "skyrmion_pattern": pattern_generator.create_skyrmion_pattern,
            "wavy_stripe_pattern": pattern_generator.create_wavy_stripe_pattern,
        }
        method = _methods.get(self.pattern_type_method)
        if method is None:
            raise ValueError(
                f"Unknown pattern_type_method: {self.pattern_type_method!r}"
            )
        return method

    def create_pattern(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate the magnetic pattern and store it on the instance.

        Physical-length values in ``pattern_config_length`` are converted to
        pixel units by dividing by ``real_space_pixel_size`` before the
        generator is called.

        Returns
        -------
        magnetic_pattern : ndarray of shape (Ny, Nx)
        pattern_coordinates : ndarray
        """
        pattern_function = self.setup()
        converted_lengths = {
            k: v / self.real_space_pixel_size
            for k, v in self.pattern_config_length.items()
        }
        self.magnetic_pattern, self.pattern_coordinates = pattern_function(
            sz_array=self.shape,
            real_space_pixel_size=self.real_space_pixel_size,
            **converted_lengths,
            **self.pattern_config,
        )
        return self.magnetic_pattern, self.pattern_coordinates

    def plot_pattern(self) -> None:
        """Display the generated pattern with real-space axes if available."""
        if self.real_space_pixel_size != 1:
            sample_y = (
                np.arange(self.shape[0]) - self.shape[0] / 2
            ) * self.real_space_pixel_size
            sample_x = (
                np.arange(self.shape[1]) - self.shape[1] / 2
            ) * self.real_space_pixel_size
            extent_real = 1e6 * np.array(
                [sample_x[0], sample_x[-1], sample_y[0], sample_y[-1]]
            )
            xlabel = "x in µm"
            ylabel = "y in µm"
        else:
            extent_real = None
            xlabel = "x in px"
            ylabel = "y in px"

        plt.figure(figsize=(5, 5))
        plt.imshow(
            self.magnetic_pattern, cmap="gray", vmin=-1, vmax=1, extent=extent_real
        )
        plt.title(self.pattern_type_method)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)


@dataclass
class FrontApertureConfig(_ConfigMixin):
    """Configuration for the front aperture (holography mask) on the sample.

    Parameters
    ----------
    aperture_method : {"FTH_circular"} or None
        Aperture layout to create. ``None`` produces a fully transparent mask.
    aperture_shape : tuple of int
        2-D array shape ``(Ny, Nx)`` in pixels.
    real_space_pixel_size : float
        Physical pixel size in metres.
    aperture_thickness : array-like of float
        Layer thicknesses of the aperture material stack in metres, passed
        directly to ``structures.Apertures3D``.
    aperture_config : dict
        Method-specific parameters. For ``"FTH_circular"`` the following keys
        are required:

        - ``apertures_type`` : list of {"OH", "RH"} — object hole or reference hole.
        - ``apertures_radius`` : list of float — radii in metres.
        - ``apertures_center`` : list of tuple — (y, x) centres in metres.
        - ``apertures_sigma`` : list of float — edge-smoothing sigma in metres.
        - ``thickness_OH`` : float — depth of the object hole in metres.

    Attributes
    ----------
    aperture : structures.Apertures3D
        The constructed aperture object, populated by ``setup()``.
    """

    aperture_method: Literal["FTH_circular"] | None = "FTH_circular"
    aperture_shape: tuple[int, int] = (256, 256)  # px
    real_space_pixel_size: float = 10e-9  # m/px
    aperture_thickness: float = 0.01  # m
    aperture_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.aperture_thickness <= 0:
            raise ValueError(
                f"aperture_thickness must be positive, got {self.aperture_thickness}"
            )

    def setup(self) -> structures.Apertures3D:
        """Build the aperture object and populate it with holes.

        Returns
        -------
        structures.Apertures3D
            The configured aperture object.
        """
        self.aperture = structures.Apertures3D(
            self.aperture_shape,
            self.real_space_pixel_size,
            layer_thicknesses=self.aperture_thickness,
        )
        if self.aperture_method == "FTH_circular":
            self.create_fth_apertures()
        elif self.aperture_method is None:
            self.aperture.create_empty_aperture()
        else:
            raise ValueError(f"Aperture method not defined, got {self.aperture_method}")

    def create_fth_apertures(self) -> None:
        """Iterate over the aperture list in ``aperture_config`` and punch holes.

        Object holes (``"OH"``) use ``thickness_OH`` as depth; reference holes
        (``"RH"``) use the full stack thickness (``sum(aperture_thickness)``).
        All coordinates are interpreted as real-space metres.
        """
        types = self.aperture_config.get("apertures_type", None)
        radi = self.aperture_config.get("apertures_radius", None)
        centers = self.aperture_config.get("apertures_center", None)
        sigmas = self.aperture_config.get("apertures_sigma", None)

        for type, radius, center, sigma in zip(types, radi, centers, sigmas):
            if type == "OH":
                depth = self.aperture_config.get("thickness_OH", None)
            elif type == "RH":
                depth = self.aperture_thickness
            else:
                raise ValueError(f"Aperture type not defined, got {type}")

            self.aperture.create_circle_aperture(
                center=center,
                depth=depth,
                radius=radius,
                sigma=sigma,
                use_real_space_coordinates=True,
            )

    def return_aperture(self) -> np.ndarray:
        """Return the 3-D aperture design array."""
        return self.aperture.aperture_design

    def visualize_aperture(self) -> None:
        """Display the depth-averaged aperture mask in pixel and real-space units."""
        mask = np.average(self.aperture.aperture_design, axis=0)
        extend_real = self.aperture.get_illumination_extent_real_space()
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(mask)
        ax[0].set_title(f"Projected {self.aperture_method} aperture in px")
        ax[1].imshow(mask, extent=1e6 * extend_real)
        ax[1].set_title(f"Projected {self.aperture_method} aperture in mm")
        ax[1].set_xlabel("x in µm")
        ax[1].set_ylabel("y in µm")


@dataclass
class IlluminationConfig(_ConfigMixin):
    """Configuration for the incident beam wavefield on the sample plane.

    Parameters
    ----------
    illumination_function : {"gaussian"} or None
        Spatial profile of the illumination. ``None`` produces a plane wave.
    illumination_center : tuple of int
        Centre of the illumination in pixels (row, col).
    illumination_config : dict
        Additional keyword arguments forwarded to the beam profile constructor
        (e.g. ``focus_distance`` and ``fwhm`` for a Gaussian beam).
    """

    XRayConfig: XRayConfig
    shape: tuple[int, int]
    real_space_pixel_size: float
    illumination_function: Literal["gaussian"] | None = "gaussian"
    illumination_config: dict = field(default_factory=dict)

    def _apply_illumination_function(self) -> None:
        """Apply the current illumination function to the existing illumination object."""
        if self.illumination_function == "gaussian":
            self.illumination.gauss_beam(**self.illumination_config)
        elif self.illumination_function in ("plane_wave", None):
            self.illumination.plane_wave(self.shape)

    def setup(self) -> None:
        """Compute the spatial wavefield and initialise Jones vectors.

        Builds the beam envelope (Gaussian or plane wave) from the current
        ``XRayConfig``. Expensive — call once. Use ``update_polarization()``
        to switch polarisation state without rebuilding the envelope.
        """
        self.beam_params = self.XRayConfig.setup()
        self.illumination = light_beam.illumination(
            self.beam_params, self.shape, self.real_space_pixel_size
        )
        self._apply_illumination_function()
        self.illumination.get_illumination_jones()

    def update_polarization(self, pol: str) -> None:
        """Switch polarisation and recompute Jones vectors without rebuilding the wavefield.

        Parameters
        ----------
        pol : {"CR", "CL", "x", "y"}
            New polarisation state.
        """
        self.XRayConfig.polarization = pol
        self.illumination.beam_parameters.pol = pol
        self.illumination.get_illumination_jones()

    def update_illumination_config(self, illumination_config: dict) -> None:
        """Update beam profile parameters and recompute the envelope and Jones vectors.

        Use when the spatial beam parameters change (e.g. ``fwhm``, ``distance``,
        ``center``) without a change in energy or polarisation.

        Parameters
        ----------
        illumination_config : dict
            New keyword arguments forwarded to ``gauss_beam``.
        """
        self.illumination_config = illumination_config
        self._apply_illumination_function()
        self.illumination.get_illumination_jones()

    def update_energy(self, energy: float) -> None:
        """Update photon energy, recompute beam envelope and Jones vectors.

        Rebuilds ``beam_params`` (new wavevector/wavelength) and re-runs
        ``gauss_beam`` if the illumination function is Gaussian, since the
        beam profile depends on wavelength. Cheaper than a full ``setup()``
        because the illumination object and its coordinate grids are reused.

        Parameters
        ----------
        energy : float
            New photon energy in eV.
        """
        self.XRayConfig.energy = energy
        self.beam_params = self.XRayConfig.setup()
        self.illumination.beam_parameters = self.beam_params
        self._apply_illumination_function()
        self.illumination.get_illumination_jones()

    def update(self, new_xray_config: XRayConfig) -> None:
        """Update illumination from a new XRayConfig, calling only what changed.

        Compares ``new_xray_config`` against the current ``self.XRayConfig``
        field by field and dispatches the minimal set of update functions.
        Replaces ``self.XRayConfig`` with ``new_xray_config`` afterwards.
        No-op if nothing changed.

        Parameters
        ----------
        new_xray_config : XRayConfig
            New X-ray source configuration to apply.
        """
        energy_changed = (
            new_xray_config.beam_params.energy != self.XRayConfig.beam_params.energy
        )
        pol_changed = new_xray_config.beam_params.pol != self.XRayConfig.beam_params.pol

        if not energy_changed and not pol_changed:
            return

        if energy_changed:
            self.update_energy(new_xray_config.energy)

        if pol_changed:
            self.update_polarization(new_xray_config.pol)

    def visualize_illumination(self) -> None:
        extend_real = self.illumination.get_illumination_extent_real_space()
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        ma = np.max(np.abs(self.illumination.illumination) ** 2)
        m0 = ax[0].imshow(
            np.abs(self.illumination.illumination) ** 2,
            extent=1e6 * extend_real,
            vmin=0,
            vmax=ma,
        )
        ax[0].set_title("Intensity")
        ax[0].set_xlabel("x in µm")
        ax[0].set_ylabel("y in µm")
        plt.colorbar(m0, ax=ax[0], label="Intensity")

        m1 = ax[1].imshow(
            np.angle(self.illumination.illumination),
            extent=1e6 * extend_real,
            vmin=-np.pi,
            vmax=np.pi,
            cmap="hsv",
        )
        ax[1].set_title("Phase")
        ax[1].set_xlabel("x in µm")
        ax[1].set_ylabel("y in µm")
        plt.colorbar(m1, ax=ax[1], label="Phase in rad")


@dataclass
class SamplePropagatorConfig(_ConfigMixin):
    """Configuration for the beam propagation through the sample structure.

    Parameters
    ----------
    SampleConfig : SampleConfig
        Sample configuration used to build the sample structure and compute the
        refractive index distribution for propagation.
    IlluminationConfig : IlluminationConfig
        Illumination configuration used to define the incident beam properties.
    propagator_method : {"Jones"} or None
        Which beam propagation method to use. ``None`` produces an empty
        propagator that returns the input wavefield unchanged.
    propagator_config : dict
        Reserved for future extension.
    """

    SampleConfig: SampleConfig
    IlluminationConfig: IlluminationConfig
    propagator_method: Literal["Jones"] | None = "Jones"
    propagator_config: dict = field(default_factory=dict)

    def setup(self) -> Jones_propagator.wavefronts:
        """Build the beam propagator object based on the current sample and illumination.

        Returns
        -------
        Jones_propagator.JonesPropagator
            The configured beam propagator object.
        """
        if self.propagator_method == "Jones":
            self.wavefront = self._jones_propagation()
        elif self.propagator_method is None:
            return None
        else:
            raise ValueError(f"Unknown propagator method: {self.propagator_method!r}")

    def _jones_propagation(self):
        wavefront = Jones_propagator.wavefronts(
            beam_parameters=self.IlluminationConfig.beam_params,
            eps_stack=self.SampleConfig.sample_structure.final_dielectric_tensor,
            layer_thicknesses=self.SampleConfig.sample_structure.layer_thicknesses,
            real_space_pixel_size=self.SampleConfig.real_space_pixel_size,
            E_in=self.IlluminationConfig.illumination.illumination_jones,
            propagate=False,
        )
        return wavefront

    def return_wavefront(self) -> Jones_propagator.wavefronts:
        """Return the configured wavefront object."""
        return self.wavefront
