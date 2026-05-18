from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from scattering_calculator.experimental_conditions import detector, light_beam
from scattering_calculator.sample_generator import pattern_generator
from scattering_calculator.sample_generator import structures
from scattering_calculator.beam_propagator import Jones_propagator


class _ConfigMixin:
    def to_dict(self) -> dict:
        """Return all configuration fields and their current values as a dict."""
        return dataclasses.asdict(self)

    def get_metadata(self, prefix: str = "") -> dict[str, int | float | str | bool]:
        """Return scalar and string configuration parameters, suitable for HDF5 attributes.

        Recursively flattens nested dicts and skips array-like values and
        nested dataclass instances.

        Parameters
        ----------
        prefix : str
            Prepended to every key, e.g. ``"detector/"`` to namespace the output.

        Returns
        -------
        dict[str, int | float | str | bool]
            Flat mapping of parameter name to scalar value.
        """
        _SCALAR_TYPES = (int, float, str, bool, np.integer, np.floating)

        def _flatten(value, key: str) -> dict:
            if value is None:
                return {key: "None"}
            if isinstance(value, _SCALAR_TYPES):
                return {key: value}
            if isinstance(value, dict):
                out = {}
                for k, v in value.items():
                    out.update(_flatten(v, f"{key}/{k}"))
                return out
            if isinstance(value, (list, tuple)) and all(
                isinstance(v, _SCALAR_TYPES) for v in value
            ):
                return {key: str(value)}
            return {}

        result: dict[str, int | float | str | bool] = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            result.update(_flatten(val, f"{prefix}{f.name}"))
        return result


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

    def get_metadata(self, prefix: str = "") -> dict[str, int | float | str | bool]:
        """Return config fields plus derived beam parameters as scalar metadata."""
        meta = super().get_metadata(prefix=prefix)
        if hasattr(self, "beam_params"):
            bp = self.beam_params
            bp_prefix = f"{prefix}beam_params/"
            for attr in ("energy", "wavelength", "wavevector", "pol", "photon_flux", "coherence_length"):
                meta[f"{bp_prefix}{attr}"] = getattr(bp, attr)
        return meta


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
            self.detector_layout.assign_beamstop(self.beamstop.return_beamstop())

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

    def visualize_beamstop(self) -> None:
        """Display the beamstop mask using the detector layout's visualizer."""
        if not hasattr(self, "beamstop"):
            raise ValueError(
                "No beamstop configured. Set beamstop_config and call setup() first."
            )
        self.detector_layout.visualize_beamstop()


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
    aperture_thicknesses: list[float] = field(default_factory=lambda: [0.01])  # m
    aperture_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(thickness <= 0 for thickness in self.aperture_thicknesses):
            raise ValueError(
                f"aperture_thickness must be positive, got {self.aperture_thicknesses}"
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
            layer_thicknesses=self.aperture_thicknesses,
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
                depth = np.sum(self.aperture_thicknesses)
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

    def calculate_scalar_wavefield(self) -> np.ndarray:
        amp = (
            np.abs(self.wavefront.exit_wave[..., 0]) ** 2
            + np.abs(self.wavefront.exit_wave[..., 1]) ** 2
        ) / np.sqrt(2)
        phase = np.angle(self.wavefront.exit_wave[..., 0])

        self.exit_wavefield = amp * np.exp(1j * phase)

    def return_scalar_wavefield(self) -> np.ndarray:
        """Return the scalar exit wavefield after sample propagation."""
        if not hasattr(self, "exit_wavefield"):
            self.calculate_scalar_wavefield()
        return self.exit_wavefield

    def visualize_exit_wavefront(self) -> None:
        """Display the intensity and phase of the exit wavefront."""
        self.calculate_scalar_wavefield()
        extend_real = (
            self.IlluminationConfig.illumination.get_illumination_extent_real_space()
        )
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        fig.suptitle(
            "Exit wavefield after sample propagation, Polarization: "
            + self.IlluminationConfig.beam_params.pol
        )
        ma = np.max(np.abs(self.exit_wavefield) ** 2)
        m0 = ax[0].imshow(
            np.abs(self.exit_wavefield) ** 2,
            extent=1e6 * extend_real,
            vmin=0,
            vmax=ma,
        )
        ax[0].set_title("Amplitude")
        ax[0].set_xlabel("x in µm")
        ax[0].set_ylabel("y in µm")
        plt.colorbar(m0, ax=ax[0], label="Amplitude")

        m1 = ax[1].imshow(
            np.angle(self.exit_wavefield),
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
class HologramConfig(_ConfigMixin):
    """Container for ideal/detected holograms and scalar exit waves, indexed by helicity.

    Parameters
    ----------
    ideal_holograms : dict[str, ndarray]
        Noise-free holograms keyed by helicity, e.g.
        ``{"CR": array_CR, "CL": array_CL}``.
        Each array may be 2-D ``(Ny, Nx)`` for a single frame or 3-D
        ``(N_frames, Ny, Nx)`` for a stack of frames.
    detected_holograms : dict[str, ndarray]
        Detected (noisy) holograms keyed by helicity, same key set as
        ``ideal_holograms``. Same shape convention as above.
    exit_waves : dict[str, ndarray]
        Scalar exit wavefields keyed by helicity, e.g.
        ``{"CR": wave_CR, "CL": wave_CL}``.
        Each array may be 2-D ``(Ny, Nx)`` (complex) or 3-D
        ``(N_frames, Ny, Nx)`` for a stack of frames.
    detector_layout : detector.detector_layout or None
        Detector geometry associated with these holograms, as returned by
        ``DetectorConfig.setup()``. Used to provide q-space / real-space
        coordinate context for downstream analysis.
    sample_x : NDArray[np.float64] or None
        2-D meshgrid of sample-plane x-coordinates in metres ``(Ny, Nx)``.
        Used to set real-space axes when visualising exit waves.
    sample_y : NDArray[np.float64] or None
        2-D meshgrid of sample-plane y-coordinates in metres ``(Ny, Nx)``.
    """

    ideal_holograms: dict = field(default_factory=dict)
    detected_holograms: dict = field(default_factory=dict)
    exit_waves: dict = field(default_factory=dict)
    detector_layout: detector.detector_layout | None = None
    sample_x: NDArray[np.float64] | None = None
    sample_y: NDArray[np.float64] | None = None

    _VALID_HELICITIES: tuple = field(
        default=("CR", "CL", "x", "y"), init=False, repr=False
    )

    def __post_init__(self) -> None:
        for store_name, store in (
            ("ideal_holograms", self.ideal_holograms),
            ("detected_holograms", self.detected_holograms),
            ("exit_waves", self.exit_waves),
        ):
            for helicity, arr in store.items():
                if helicity not in self._VALID_HELICITIES:
                    raise ValueError(
                        f"Invalid helicity {helicity!r} in {store_name}. "
                        f"Must be one of {self._VALID_HELICITIES}."
                    )
                if arr.ndim < 2:
                    raise ValueError(
                        f"{store_name}[{helicity!r}] must be 2-D or 3-D, "
                        f"got shape {arr.shape}."
                    )

    def _get_store(self, source: Literal["ideal", "detected", "exit_wave"]) -> dict:
        if source == "ideal":
            return self.ideal_holograms
        if source == "detected":
            return self.detected_holograms
        if source == "exit_wave":
            return self.exit_waves
        raise ValueError(
            f"Unknown source {source!r}. Must be 'ideal', 'detected', or 'exit_wave'."
        )

    @property
    def helicities(self) -> list[str]:
        """Helicity keys present in the ideal hologram dict."""
        return list(self.ideal_holograms.keys())

    def add_ideal_hologram(self, helicity: str, hologram: np.ndarray) -> None:
        """Add or replace an ideal hologram for *helicity*.

        *hologram* may be 2-D ``(Ny, Nx)`` or 3-D ``(N_frames, Ny, Nx)``.
        """
        if helicity not in self._VALID_HELICITIES:
            raise ValueError(
                f"Invalid helicity {helicity!r}. Must be one of {self._VALID_HELICITIES}."
            )
        if hologram.ndim < 2:
            raise ValueError(
                f"hologram must be 2-D or 3-D, got shape {hologram.shape}."
            )
        self.ideal_holograms[helicity] = hologram

    def add_detected_hologram(self, helicity: str, hologram: np.ndarray) -> None:
        """Add or replace a detected hologram for *helicity*.

        *hologram* may be 2-D ``(Ny, Nx)`` or 3-D ``(N_frames, Ny, Nx)``.
        """
        if helicity not in self._VALID_HELICITIES:
            raise ValueError(
                f"Invalid helicity {helicity!r}. Must be one of {self._VALID_HELICITIES}."
            )
        if hologram.ndim < 2:
            raise ValueError(
                f"hologram must be 2-D or 3-D, got shape {hologram.shape}."
            )
        self.detected_holograms[helicity] = hologram

    def _stack_into(self, store: dict, data: dict) -> None:
        """Concatenate *data* arrays into *store* along axis 0."""
        for helicity, arr in data.items():
            if helicity not in self._VALID_HELICITIES:
                raise ValueError(
                    f"Invalid helicity {helicity!r}. Must be one of {self._VALID_HELICITIES}."
                )
            if arr.ndim < 2:
                raise ValueError(
                    f"Array for helicity {helicity!r} must be 2-D or 3-D, "
                    f"got shape {arr.shape}."
                )
            new = arr[np.newaxis] if arr.ndim == 2 else arr
            if helicity not in store:
                store[helicity] = new
            else:
                existing = store[helicity]
                if existing.ndim == 2:
                    existing = existing[np.newaxis]
                store[helicity] = np.concatenate([existing, new], axis=0)

    def add_holograms(
        self,
        holograms: dict,
        source: Literal["ideal", "detected"] = "detected",
    ) -> None:
        """Stack new holograms onto the existing store for each helicity.

        Parameters
        ----------
        holograms : dict[str, ndarray]
            Mapping of helicity → array. Each array may be 2-D ``(Ny, Nx)``
            or 3-D ``(N_frames, Ny, Nx)``.
        source : {"ideal", "detected"}
            Which hologram store to append to.
        """
        self._stack_into(self._get_store(source), holograms)

    def add_exit_waves(self, exit_waves: dict) -> None:
        """Stack new scalar exit wavefields onto the exit-wave store.

        Parameters
        ----------
        exit_waves : dict[str, ndarray]
            Mapping of helicity → complex array. Each array may be 2-D
            ``(Ny, Nx)`` or 3-D ``(N_frames, Ny, Nx)``.
        """
        self._stack_into(self.exit_waves, exit_waves)

    def average_stack(
        self, source: Literal["ideal", "detected", "exit_wave"] = "detected"
    ) -> dict[str, np.ndarray]:
        """Average frame stacks along axis 0 for each helicity.

        2-D arrays are returned unchanged. 3-D arrays of shape
        ``(N_frames, Ny, Nx)`` are reduced to ``(Ny, Nx)`` by mean.

        Parameters
        ----------
        source : {"ideal", "detected", "exit_wave"}
            Which store to average.

        Returns
        -------
        dict[str, ndarray]
            New dict with the same helicity keys and 2-D averaged arrays.
        """
        store = self._get_store(source)
        return {
            h: arr.mean(axis=0) if arr.ndim == 3 else arr for h, arr in store.items()
        }

    def difference(
        self, source: Literal["ideal", "detected", "exit_wave"] = "detected"
    ) -> np.ndarray:
        """Compute CR − CL, store as ``"diff"`` in the source store, and return it.

        Parameters
        ----------
        source : {"ideal", "detected", "exit_wave"}
            Which store to use.
        """
        store = self._get_store(source)
        if "CR" not in store or "CL" not in store:
            raise ValueError(
                f"Both 'CR' and 'CL' entries are required for a difference "
                f"(source={source!r})."
            )
        store["diff"] = store["CR"] - store["CL"]
        return store["diff"]

    def sum(
        self, source: Literal["ideal", "detected", "exit_wave"] = "detected"
    ) -> np.ndarray:
        """Compute CR + CL, store as ``"sum"`` in the source store, and return it.

        Parameters
        ----------
        source : {"ideal", "detected", "exit_wave"}
            Which store to use.
        """
        store = self._get_store(source)
        if "CR" not in store or "CL" not in store:
            raise ValueError(
                f"Both 'CR' and 'CL' entries are required for a sum "
                f"(source={source!r})."
            )
        store["sum"] = store["CR"] + store["CL"]
        return store["sum"]

    def compute_differences(self):
        for source in ["ideal", "detected", "exit_wave"]:
            self.difference(source)

    def compute_sums(self):
        for source in ["ideal", "detected", "exit_wave"]:
            self.sum(source)

    def compute_reconstructions(self):
        for source in ["ideal", "detected", "exit_wave"]:
            store = self._get_store(source)
            for helicity in list(store.keys()):
                self.reconstruct(source, helicity)

    @staticmethod
    def _fth_reconstruct(holo: np.ndarray) -> np.ndarray:
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(holo)))

    def reconstruct(
        self,
        source: Literal["ideal", "detected", "exit_wave"] = "ideal",
        helicity: str = "diff",
    ) -> np.ndarray:
        """Apply FTH reconstruction to a hologram and store the result.

        The reconstruction is ``fftshift(fft2(fftshift(holo)))``.
        Results are accumulated in ``self.reconstructions[source][helicity]``.

        Parameters
        ----------
        source : {"ideal", "detected", "exit_wave"}
            Which store to read from.
        helicity : str
            Key within that store, e.g. ``"diff"``, ``"sum"``, ``"CR"``.

        Returns
        -------
        ndarray of shape (Ny, Nx), complex
            The reconstructed image.
        """
        store = self._get_store(source)
        if helicity not in store:
            raise ValueError(
                f"{helicity!r} not found in {source!r} store. "
                f"Available keys: {list(store.keys())}."
            )
        result = self._fth_reconstruct(store[helicity])
        if not hasattr(self, "reconstructions"):
            self.reconstructions: dict = {}
        self.reconstructions.setdefault(source, {})[helicity] = result
        return result

    def visualize_reconstruction(
        self,
        source: Literal["ideal", "detected", "exit_wave"] = "detected",
        helicity: str = "diff",
        frame: int | None = None,
    ) -> None:
        """Display abs, phase, real, and imaginary parts of a reconstruction.

        By default (``frame=None``) the averaged hologram is reconstructed.
        Pass a frame index to reconstruct a specific slice from the raw stack.

        Parameters
        ----------
        source : {"ideal", "detected", "exit_wave"}
            Which store to reconstruct from.
        helicity : str
            Key within that store, e.g. ``"diff"``, ``"sum"``, ``"CR"``.
        frame : int or None
            If ``None`` (default), use the frame-averaged hologram.
            If an integer, select that frame from the raw 3-D stack
            ``(N_frames, Ny, Nx)`` stored in the source store.
        """
        if frame is None:
            if not hasattr(self, "exit_waves_avg"):
                self.compute_averages()
            avg_store = {
                "ideal": self.ideal_holograms_avg,
                "detected": self.detected_holograms_avg,
                "exit_wave": self.exit_waves_avg,
            }[source]
            if helicity not in avg_store:
                raise ValueError(
                    f"{helicity!r} not found in averaged {source!r} store. "
                    f"Available keys: {list(avg_store.keys())}."
                )
            holo = avg_store[helicity]
            frame_label = "averaged"
        else:
            raw_store = self._get_store(source)
            if helicity not in raw_store:
                raise ValueError(
                    f"{helicity!r} not found in {source!r} store. "
                    f"Available keys: {list(raw_store.keys())}."
                )
            arr = raw_store[helicity]
            if arr.ndim != 3:
                raise ValueError(
                    f"{source!r}[{helicity!r}] is not a stack (shape {arr.shape}). "
                    "Pass frame=None to use the 2-D array directly."
                )
            if not (0 <= frame < arr.shape[0]):
                raise IndexError(
                    f"frame={frame} out of range for stack of {arr.shape[0]} frames."
                )
            holo = arr[frame]
            frame_label = f"frame {frame}"

        rec = self._fth_reconstruct(holo)

        panels = [
            (np.abs(rec), "Amplitude", "inferno", None, None, "Amplitude"),
            (np.angle(rec), "Phase", "hsv", -np.pi, np.pi, "Phase in rad"),
            (np.real(rec), "Real", "gray", None, None, "Real part"),
            (np.imag(rec), "Imaginary", "gray", None, None, "Imag part"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(
            f"Reconstruction — source: {source!r}, helicity: {helicity!r}, {frame_label}"
        )

        for ax, (data, title, cmap, vmin, vmax, clabel) in zip(axes.flat, panels):
            if vmin is None:
                vmin, vmax = np.nanpercentile(data, [1, 99])
            m = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("x in px")
            ax.set_ylabel("y in px")
            fig.colorbar(m, ax=ax, label=clabel)

    def compute_averages(self) -> None:
        """Average all frame stacks and store results as instance attributes.

        Populates:

        - ``ideal_holograms_avg``   — averaged ideal holograms per helicity
        - ``detected_holograms_avg`` — averaged detected holograms per helicity
        - ``exit_waves_avg``         — averaged exit wavefields per helicity

        3-D arrays ``(N_frames, Ny, Nx)`` are reduced to ``(Ny, Nx)`` by
        mean along axis 0. 2-D arrays are stored unchanged.
        """
        self.ideal_holograms_avg = self.average_stack("ideal")
        self.detected_holograms_avg = self.average_stack("detected")
        self.exit_waves_avg = self.average_stack("exit_wave")

    def visualize_averages(self) -> None:
        """Display averaged exit waves and holograms for the first two helicities.

        Calls ``compute_averages()`` first if the averaged attributes are not
        yet present.

        Layout (4 rows × 2 columns):

        - Row 1: Amplitude | Phase  of averaged exit wave — helicity 1
        - Row 2: Amplitude | Phase  of averaged exit wave — helicity 2
        - Row 3: Ideal     | Detected hologram            — helicity 1
        - Row 4: Ideal     | Detected hologram            — helicity 2
        """
        if not hasattr(self, "exit_waves_avg"):
            self.compute_averages()

        helicities = list(self.exit_waves_avg.keys())
        if len(helicities) < 2:
            raise ValueError(
                f"At least two helicities are required for this plot, "
                f"got {helicities}."
            )
        h1, h2 = helicities[:2]

        # sample-plane extent in µm (exit wave rows)
        if self.sample_x is not None and self.sample_y is not None:
            sx, sy = self.sample_x, self.sample_y
            sample_extent = 1e6 * np.array([sx[0, 0], sx[0, -1], sy[0, 0], sy[-1, 0]])
            sample_xlabel = sample_ylabel = "µm"
        else:
            sample_extent = None
            sample_xlabel = sample_ylabel = "px"

        # detector q-space extent in nm⁻¹ (hologram rows)
        if self.detector_layout is not None and hasattr(self.detector_layout, "detqx"):
            qx = self.detector_layout.detqx
            qy = self.detector_layout.detqy
            det_extent = 1e-9 * np.array([qx[0, 0], qx[0, -1], qy[0, 0], qy[-1, 0]])
            det_xlabel = det_ylabel = "nm⁻¹"
        else:
            det_extent = None
            det_xlabel = det_ylabel = "px"

        fig, axes = plt.subplots(4, 2, figsize=(10, 16))
        fig.suptitle("Averaged exit waves and holograms")

        for row, helicity in enumerate([h1, h2]):
            wave = self.exit_waves_avg[helicity]
            amp = np.abs(wave)
            phase = np.angle(wave)

            m0 = axes[row, 0].imshow(amp, cmap="inferno", extent=sample_extent)
            axes[row, 0].set_title(f"Exit wave amplitude — {helicity}")
            axes[row, 0].set_xlabel(f"x in {sample_xlabel}")
            axes[row, 0].set_ylabel(f"y in {sample_ylabel}")
            fig.colorbar(m0, ax=axes[row, 0], label="Amplitude")

            m1 = axes[row, 1].imshow(
                phase, cmap="hsv", vmin=-np.pi, vmax=np.pi, extent=sample_extent
            )
            axes[row, 1].set_title(f"Exit wave phase — {helicity}")
            axes[row, 1].set_xlabel(f"x in {sample_xlabel}")
            axes[row, 1].set_ylabel(f"y in {sample_ylabel}")
            fig.colorbar(m1, ax=axes[row, 1], label="Phase in rad")

        for row, helicity in enumerate([h1, h2], start=2):
            ideal = self.ideal_holograms_avg[helicity]
            detected = self.detected_holograms_avg[helicity]

            ideal_vmin, ideal_vmax = np.nanpercentile(ideal, [0.1, 99.9])
            det_vmin, det_vmax = np.nanpercentile(detected, [0.1, 99.9])

            m2 = axes[row, 0].imshow(
                ideal,
                cmap="viridis",
                vmin=ideal_vmin,
                vmax=ideal_vmax,
                extent=det_extent,
            )
            axes[row, 0].set_title(f"Ideal hologram — {helicity}")
            axes[row, 0].set_xlabel(f"x in {det_xlabel}")
            axes[row, 0].set_ylabel(f"y in {det_ylabel}")
            fig.colorbar(m2, ax=axes[row, 0], label="counts")

            m3 = axes[row, 1].imshow(
                detected,
                cmap="viridis",
                vmin=det_vmin,
                vmax=det_vmax,
                extent=det_extent,
            )
            axes[row, 1].set_title(f"Detected hologram — {helicity}")
            axes[row, 1].set_xlabel(f"x in {det_xlabel}")
            axes[row, 1].set_ylabel(f"y in {det_ylabel}")
            fig.colorbar(m3, ax=axes[row, 1], label="counts")

    def to_dict(
        self,
        helicities: list[str] | None = None,
        sources: list[Literal["exit_wave", "ideal", "detected"]] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return exit waves, ideal holograms, and detected holograms grouped by helicity.

        Parameters
        ----------
        helicities : list of str or None
            Helicity keys to include, e.g. ``["CR", "CL"]``.
            If ``None`` (default), all available helicities are returned.
        sources : list of {"exit_wave", "ideal", "detected"} or None
            Which data stores to include. If ``None`` (default), all three are returned.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            Outer key: helicity (e.g. ``"CR"``, ``"CL"``).
            Inner keys: ``"exit_wave"``, ``"ideal"``, ``"detected"``.
            Missing entries for a given helicity are omitted.
        """
        _store_map = {
            "exit_wave": self.exit_waves,
            "ideal": self.ideal_holograms,
            "detected": self.detected_holograms,
        }
        active_sources = set(sources) if sources is not None else set(_store_map)

        available = set().union(*(s.keys() for s in _store_map.values()))
        keys = set(helicities) if helicities is not None else available

        result: dict[str, dict[str, np.ndarray]] = {}
        for h in keys:
            entry: dict[str, np.ndarray] = {}
            for source in active_sources:
                store = _store_map[source]
                if h in store:
                    entry[source] = store[h]
            result[h] = entry
        return result
