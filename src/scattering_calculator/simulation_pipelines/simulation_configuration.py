"""Simulation-configuration dataclasses for scattering workflows."""

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
        """Return all configuration fields and their current values as a dict.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : dict
            Return value produced by the function.
        """
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
            """Handle the internal flatten operation.

            Parameters
            ----------
            value : Any
                Input value for ``value``.
            key : str
                Input value for ``key``.

            Returns
            -------
            result : dict
                Return value produced by the function.
            """
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
    coherence_length : tuple of float
        Transverse coherence length in metres as ``(y, x)``.

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
    coherence_length: tuple[float, float] = (10e-6, 10e-6)  # m, (y, x)

    def __post_init__(self) -> None:
        """Handle the internal post init operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        if self.energy <= 0:
            raise ValueError(f"energy must be positive, got {self.energy}")
        if self.photon_flux <= 0:
            raise ValueError(f"photon_flux must be positive, got {self.photon_flux}")
        if np.isscalar(self.coherence_length):
            self.coherence_length = (
                float(self.coherence_length),
                float(self.coherence_length),
            )
        if len(self.coherence_length) != 2:
            raise ValueError(
                "coherence_length must be a scalar or a 2-tuple "
                "(coherence_length_y, coherence_length_x)."
            )
        if any(length <= 0 for length in self.coherence_length):
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

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """
        self.beam_params = light_beam.beam_parameters(
            self.energy, self.pol, self.photon_flux, self.coherence_length
        )
        self.beam_params.calc_wavevector()
        return self.beam_params

    def get_metadata(self, prefix: str = "") -> dict[str, int | float | str | bool]:
        """Return derived beam parameters as scalar metadata.

        Parameters
        ----------
        prefix : str
            Input value for ``prefix``.

        Returns
        -------
        result : dict[str, int | float | str | bool]
            Return value produced by the function.
        """
        meta: dict[str, int | float | str | bool] = {}
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
        Optional keyword arguments forwarded to the beamstop creation method.
        Missing values use the detector defaults. If ``"radius"`` is missing,
        no beamstop or wire is created. Lengths, including ``"sigma"``, are in
        metres when the beamstop is created through this config.
    """

    bs_method: Literal["circular", None] | None = "circular"
    bs_detector_distance: float = 0.01  # m
    bs_center: tuple[int, int] = (0, 0)  # px
    bs_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Handle the internal post init operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        if self.bs_detector_distance <= 0:
            raise ValueError(
                f"bs_detector_distance must be positive, got {self.bs_detector_distance}"
            )
        radius = self.bs_config.get("radius")
        if radius is not None and radius <= 0:
            raise ValueError(f'bs_config["radius"] must be positive, got {radius}')
        sigma = self.bs_config.get("sigma")
        if sigma is not None and sigma < 0:
            raise ValueError(f'bs_config["sigma"] must be non-negative, got {sigma}')
        ellipticity = self.bs_config.get(
            "ellipticity", self.bs_config.get("ellipticity_range", (1.0, 1.0))
        )
        if any(v <= 0 for v in ellipticity):
            raise ValueError(
                'bs_config["ellipticity"] values must be positive, '
                f"got {ellipticity}"
            )
        if ellipticity[0] > ellipticity[1]:
            raise ValueError(
                'bs_config["ellipticity"] must be ordered as (min, max), '
                f"got {ellipticity}"
            )
        roughness = self.bs_config.get("roughness", 0.0)
        if roughness < 0:
            raise ValueError(
                f'bs_config["roughness"] must be non-negative, got {roughness}'
            )
        roughness_modes = self.bs_config.get(
            "roughness_modes", self.bs_config.get("rougness_modes", (0, 0))
        )
        if roughness > 0 and roughness_modes[0] < 1:
            raise ValueError(
                'bs_config["roughness_modes"] must start at 1 or greater when '
                f'bs_config["roughness"] is positive, got {roughness_modes}'
            )
        if roughness_modes[0] > roughness_modes[1]:
            raise ValueError(
                'bs_config["roughness_modes"] must be ordered as (min, max), '
                f"got {roughness_modes}"
            )
        wire_width = self.bs_config.get("wire_width", 0.0)
        if wire_width < 0:
            raise ValueError(
                f'bs_config["wire_width"] must be non-negative, got {wire_width}'
            )
        wire_bend = self.bs_config.get("wire_bend", 0.0)
        if wire_bend < 0:
            raise ValueError(
                f'bs_config["wire_bend"] must be non-negative, got {wire_bend}'
            )
        antialias = self.bs_config.get("antialias", 1)
        if int(antialias) < 1:
            raise ValueError(
                f'bs_config["antialias"] must be at least 1, got {antialias}'
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
            bs.create_empty_beamstop()
            return bs
        if self.bs_method == "circular":
            beamstop_kwargs = dict(self.bs_config)
            if "rougness_modes" in beamstop_kwargs:
                beamstop_kwargs["roughness_modes"] = beamstop_kwargs.pop(
                    "rougness_modes"
                )
            bs.create_circle_beamstop(
                center=self.bs_center,
                use_real_space_coordinates=True,
                **beamstop_kwargs,
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
    use_detector_pixel_footprint : bool
        If ``True``, project ideal holograms by averaging over a sub-sampling
        grid inside each detector pixel. If ``False``, sample at detector pixel
        centres using positivity-preserving linear interpolation.
    detector_pixel_footprint_samples : int
        Number of sub-samples per detector-pixel axis when
        ``use_detector_pixel_footprint`` is ``True``.

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
        default_factory=lambda: {
            "readout_noise_average": 50,
            "readout_noise_sigma": 3,
            "detector_threshold": 64e3,
            "counts_per_photon": 100,
            "quantum_efficiency": 1.0,
        }
    )
    artifacts_method: str | None = None
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
    beamstop_config: BeamstopConfig | None = None
    use_detector_pixel_footprint: bool = False
    detector_pixel_footprint_samples: int = 3

    def __post_init__(self) -> None:
        """Handle the internal post init operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
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

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
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
        """Assign a simulated hologram to the detector layout for later retrieval.

        Parameters
        ----------
        samplepropagationconfig : Any
            Input value for ``samplepropagationconfig``.

        Returns
        -------
        None
            The function completes in place.
        """
        self.propagator = samplepropagationconfig
        self.wavefront = self.propagator.return_wavefront()
        exit_wavefield = self.propagator.return_scalar_wavefield()
        farfield_exit_wave = getattr(
            self.wavefront,
            "exit_wave_for_farfield",
            self.wavefront.exit_wave,
        )
        if farfield_exit_wave is self.wavefront.exit_wave:
            exit_intensity = np.sum(np.abs(exit_wavefield) ** 2)
        else:
            exit_intensity = np.sum(np.abs(farfield_exit_wave) ** 2)
        hologram_intensity = np.sum(self.wavefront.hologram)
        if hologram_intensity <= 0:
            raise ValueError("Cannot scale detector hologram with zero total intensity.")
        self.wavefront.hologram *= exit_intensity / hologram_intensity
        self.exit_wavefield_intensity = exit_intensity
        self.hologram_intensity = np.sum(self.wavefront.hologram)

    def detect_hologram(self) -> np.ndarray:
        """Simulate the detection of the hologram on the detector, including noise and artifacts.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : np.ndarray
            Return value produced by the function.
        """
        if not hasattr(self, "wavefront"):
            raise ValueError("No propagated wavefront assigned to detector layout.")
        self.hologram_exp = detector.detector_hologram(
            self.detector_layout,
            self.wavefront.hologram,
            self.propagator.IlluminationConfig.beam_params,
            self.propagator.SampleConfig.real_space_pixel_size,
            self.beamstop,
            artifacts_config=self.artifacts_config,
            measurement_config=self.measurement_config,
            detector_params=self.detector_params,
            coherence_length=(
                self.propagator.IlluminationConfig.beam_params.coherence_length
            ),
        )
        self.hologram_exp.gnomonic_projection(
            use_pixel_footprint=self.use_detector_pixel_footprint,
            pixel_footprint_samples=self.detector_pixel_footprint_samples,
        )

    def return_ideal_hologram(self) -> np.ndarray:
        """Return the ideal (noise-free, artifact-free) hologram as a 2-D array.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : np.ndarray
            Return value produced by the function.
        """
        if not hasattr(self, "wavefront"):
            raise ValueError("No propagated wavefront assigned to detector layout.")
        return self.hologram_exp.hologram_detector

    def return_detected_hologram(
        self,
        apply_beamstop_mask: bool = True,
        apply_detector_threshold: bool = True,
        store_no_beamstop: bool = False,
    ) -> np.ndarray:
        """Return the simulated detected hologram as a 2-D array.

        Parameters
        ----------
        apply_beamstop_mask : bool
            Input value for ``apply_beamstop_mask``.
        apply_detector_threshold : bool
            Input value for ``apply_detector_threshold``.
        store_no_beamstop : bool
            Input value for ``store_no_beamstop``.

        Returns
        -------
        result : np.ndarray
            Return value produced by the function.
        """
        self.hologram_exp.add_noise(
            apply_beamstop_mask=apply_beamstop_mask,
            apply_detector_threshold=apply_detector_threshold,
            store_no_beamstop=store_no_beamstop,
        )
        if not hasattr(self, "hologram_exp"):
            raise ValueError("No hologram detected yet. Call detect_hologram() first.")
        return self.hologram_exp.hologram_exp

    def return_detected_hologram_without_beamstop(self) -> np.ndarray:
        """Return the no-beamstop detected hologram from the latest noise draw.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : np.ndarray
            Return value produced by the function.
        """
        if not hasattr(self.hologram_exp, "hologram_exp_no_beamstop"):
            raise ValueError(
                "No no-beamstop hologram stored. Call return_detected_hologram("
                "store_no_beamstop=True) first."
            )
        return self.hologram_exp.hologram_exp_no_beamstop

    def visualize_beamstop(self) -> None:
        """Display the beamstop mask using the detector layout's visualizer.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
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
        """Handle the internal post init operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        if any(s <= 0 for s in self.shape):
            raise ValueError(f"shape dimensions must be positive, got {self.shape}")
        if self.real_space_pixel_size <= 0:
            raise ValueError(
                f"real_space_pixel_size must be positive, got {self.real_space_pixel_size}"
            )

    def setup(self) -> None:
        """Build centred real-space coordinate grids and store them as ``xgrid``/``ygrid``.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
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
        Thicknesses are in nanometres.
        Slash-separated terms are separate layers; adjacent terms such as
        ``"Pt(4)Co(6)"`` are combined into one thickness-weighted
        effective-medium layer.
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
        """Parse the recipe, load refractive indices, and build the layer stack.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        self.multilayer_recipe = structures.parse_recipe(
            self.recipe,
            sample_name=self.sample_name,
            comments=self.comments,
        )

        self.material_params = structures.material_params(
            materials=set(
                material
                for layer in self.multilayer_recipe.layers
                for material, _ in (
                    layer.components or ((layer.material, layer.thickness),)
                )
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
            if layer.is_composite:
                self.sample_structure.add_effective_layer(
                    layer.material,
                    layer.components,
                )
            else:
                self.sample_structure.add_layer(layer.material, thickness=layer.thickness)

    def assign_magnetic_pattern(self, magnetic_vector_field: np.ndarray) -> None:
        """Map a 2-D scalar magnetization pattern onto the 3-D sample stack.

        Parameters
        ----------
        magnetic_vector_field : ndarray of shape (nr_layer,mx, my, mz)
            (4-D) 3-D magnetic vector field.

        Returns
        -------
        None
            The function completes in place.
        """
        self.sample_structure.magnetization = magnetic_vector_field

    def assign_aperture_mask(self, aperture_mask: np.ndarray) -> None:
        """Attach a 3-D aperture mask and store its 2-D projection.

        Parameters
        ----------
        aperture_mask : ndarray of shape (Nz, Ny, Nx)
            3-D binary/soft aperture mask.

        Returns
        -------
        None
            The function completes in place.
        """
        self.sample_structure.mask = aperture_mask
        self.sample_structure.aperture_mask2D = np.average(aperture_mask, axis=0)


@dataclass
class MagneticPatternConfig(_ConfigMixin):
    """Configuration for generating a 2-D magnetic domain pattern.

    Parameters
    ----------
    pattern_type_method : {"skyrmion_pattern", "disordered_skyrmion_lattice_pattern", "wavy_stripe_pattern", "binary_labyrinth_pattern", "saturated_pattern", "image_pattern"}
        Which pattern generator function to use.
    shape : tuple of int or None
        2-D array shape ``(Ny, Nx)`` in pixels.
    real_space_pixel_size : float or None
        Physical pixel size in metres. Used to convert physical-length entries
        in ``pattern_config`` to pixel units before calling the generator.
    pattern_config : dict
        Pattern parameters forwarded to the generator. Physical-length entries
        are specified in metres and converted to pixels automatically for the
        selected pattern type. ``sigma`` is converted for every generator that
        supports Gaussian domain-wall smoothing, including both skyrmion
        generators.
    pattern_config_length : dict
        Deprecated compatibility dict for physical-length parameters in metres.
        Values are merged into ``pattern_config`` unless the same key is already
        present there.

    Attributes
    ----------
    magnetic_pattern : ndarray of shape (Ny, Nx)
        Out-of-plane magnetization map, populated by ``create_pattern()``.
    pattern_coordinates : ndarray
        Auxiliary coordinate output from the generator, populated by
        ``create_pattern()``.
    """

    pattern_type_method: Literal[
        "skyrmion_pattern",
        "disordered_skyrmion_lattice_pattern",
        "wavy_stripe_pattern",
        "binary_labyrinth_pattern",
        "saturated_pattern",
        "image_pattern",
    ] = "skyrmion_pattern"
    shape: tuple[int, int] | None = None
    real_space_pixel_size: float | None = None
    pattern_config: dict = field(default_factory=dict)
    pattern_config_length: dict = field(default_factory=dict)

    def setup(self):
        """Return the generator function selected by ``pattern_type_method``.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        _methods = {
            "skyrmion_pattern": pattern_generator.create_skyrmion_pattern,
            "disordered_skyrmion_lattice_pattern": pattern_generator.create_disordered_skyrmion_lattice_pattern,
            "wavy_stripe_pattern": pattern_generator.create_wavy_stripe_pattern,
            "binary_labyrinth_pattern": pattern_generator.create_binary_labyrinth_pattern,
            "saturated_pattern": pattern_generator.create_saturated_pattern,
            "image_pattern": pattern_generator.create_image_pattern,
        }
        method = _methods.get(self.pattern_type_method)
        if method is None:
            raise ValueError(
                f"Unknown pattern_type_method: {self.pattern_type_method!r}"
            )
        return method

    def create_pattern(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate the magnetic pattern and store it on the instance.

        Physical-length values in ``pattern_config`` are converted to pixel
        units by dividing by ``real_space_pixel_size`` before the generator is
        called. ``pattern_config_length`` is still accepted as a legacy source,
        but conflicting keys are rejected.

        Returns
        -------
        magnetic_pattern : ndarray of shape (Ny, Nx)
        pattern_coordinates : ndarray

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """
        pattern_function = self.setup()
        config = dict(self.pattern_config)
        for key, value in self.pattern_config_length.items():
            if key in config and not np.array_equal(config[key], value):
                raise ValueError(
                    f"Conflicting magnetic-pattern value for {key!r}: use only "
                    "pattern_config. pattern_config_length is a legacy alias."
                )
            config[key] = value
        length_keys_by_method = {
            "wavy_stripe_pattern": {
                "stripe_width",
                "sigma",
                "waviness_amplitude",
                "waviness_scale",
            },
            "skyrmion_pattern": {
                "skyr_radius",
                "screening_radius",
                "sigma",
            },
            "disordered_skyrmion_lattice_pattern": {
                "stripe_width",
                "sigma",
                "diameter_spread",
                "positional_disorder",
                "placement_radius",
            },
            "binary_labyrinth_pattern": {
                "stripe_width",
                "sigma",
            },
            "image_pattern": {
                "sigma",
            },
        }
        length_keys = length_keys_by_method.get(self.pattern_type_method, set())
        generator_config = {
            k: (v / self.real_space_pixel_size if k in length_keys else v)
            for k, v in config.items()
        }
        self.magnetic_pattern, self.pattern_coordinates = pattern_function(
            sz_array=self.shape,
            real_space_pixel_size=self.real_space_pixel_size,
            **generator_config,
        )
        return self.magnetic_pattern, self.pattern_coordinates

    def plot_pattern(self) -> None:
        """Display the generated pattern with real-space axes if available.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
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
        - ``apertures_angle`` : list of float — ellipse orientation in radians.
        - ``apertures_ellipticity`` : list of float — y/x axis ratio.
        - ``apertures_roughness`` : list of float — boundary roughness amplitude.
        - ``apertures_roughness_modes`` : list of tuple — inclusive Fourier-mode range.
        - ``apertures_seed`` : list of int — random seeds for rough boundaries.
        - ``apertures_top_radius_factor`` : list of float — top/base radius
          ratio for conical holes. Default ``2``.
        - ``aperture_taper_depth`` : float, optional — depth over which holes
          taper from the wider top opening to the base radius. Defaults to
          ``thickness_OH`` so the top metal stack is conical and deeper layers
          stay cylindrical.
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
    use_roi: bool = True

    def __post_init__(self) -> None:
        """Handle the internal post init operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
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

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
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

        Object holes (``"OH"``) use ``thickness_OH`` as depth. Reference holes
        (``"RH"``) and slits (``"SLIT"``) use the full stack thickness
        (``sum(aperture_thickness)``). All coordinates are interpreted as
        real-space metres.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        types = self.aperture_config.get("apertures_type", None)
        radi = self.aperture_config.get("apertures_radius", None)
        centers = self.aperture_config.get("apertures_center", None)
        sigmas = self.aperture_config.get("apertures_sigma", None)
        angles = self._aperture_values("apertures_angle", len(types), 0.0)
        ellipticities = self._aperture_values("apertures_ellipticity", len(types), 1.0)
        roughnesses = self._aperture_values("apertures_roughness", len(types), 0.0)
        roughness_modes = self._aperture_values(
            "apertures_roughness_modes", len(types), (0, 0)
        )
        seeds = self._aperture_values("apertures_seed", len(types), None)
        top_radius_factors = self._aperture_values(
            "apertures_top_radius_factor", len(types), 2.0
        )
        lengths = self._aperture_values("apertures_length", len(types), None)

        for (
            type,
            radius,
            center,
            sigma,
            angle,
            ellipticity,
            roughness,
            modes,
            seed,
            top_radius_factor,
            length,
        ) in zip(
            types,
            radi,
            centers,
            sigmas,
            angles,
            ellipticities,
            roughnesses,
            roughness_modes,
            seeds,
            top_radius_factors,
            lengths,
        ):
            if type == "OH":
                depth = self.aperture_config.get("thickness_OH", None)
            elif type in ("RH", "SLIT"):
                depth = np.sum(self.aperture_thicknesses)
            else:
                raise ValueError(f"Aperture type not defined, got {type}")

            seed = None if seed is None or int(seed) < 0 else int(seed)
            taper_depth = self.aperture_config.get(
                "aperture_taper_depth",
                self.aperture_config.get("thickness_OH", depth),
            )

            if type == "SLIT":
                if length is None:
                    raise ValueError("SLIT apertures require apertures_length.")
                self.aperture.create_slit_aperture(
                    center=center,
                    depth=depth,
                    width=radius,
                    length=length,
                    sigma=sigma,
                    angle=angle,
                    roughness=roughness,
                    roughness_modes=tuple(modes),
                    seed=seed,
                    use_real_space_coordinates=True,
                    use_roi=self.use_roi,
                    top_radius_factor=top_radius_factor,
                    taper_depth=taper_depth,
                )
            else:
                self.aperture.create_circle_aperture(
                    center=center,
                    depth=depth,
                    radius=radius,
                    sigma=sigma,
                    angle=angle,
                    ellipticity=ellipticity,
                    roughness=roughness,
                    roughness_modes=tuple(modes),
                    seed=seed,
                    use_real_space_coordinates=True,
                    use_roi=self.use_roi,
                    top_radius_factor=top_radius_factor,
                    taper_depth=taper_depth,
                )

    def _aperture_values(self, key: str, n: int, default):
        """Return a per-aperture list, using ``default`` when absent.

        Parameters
        ----------
        key : str
            Input value for ``key``.
        n : int
            Input value for ``n``.
        default : Any
            Input value for ``default``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        values = self.aperture_config.get(key, None)
        if values is None:
            return [default] * n
        if len(values) != n:
            raise ValueError(
                f"{key} must have {n} entries, matching apertures_type; "
                f"got {len(values)}"
            )
        return values

    def return_aperture(self) -> np.ndarray:
        """Return the 3-D aperture design array.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : np.ndarray
            Return value produced by the function.
        """
        return self.aperture.aperture_design

    def create_supportmask(
        self,
        output_shape: tuple[int, int] | None = None,
        output_pixel_size: float | None = None,
        aperture_types: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        """Return a binary 2-D support mask covering all OH/RH aperture holes.

        Parameters
        ----------
        output_shape : tuple of int or None
            Shape of the returned mask. ``None`` uses the aperture/sample shape.
        output_pixel_size : float or None
            Pixel size of the returned mask in metres. ``None`` uses the aperture
            pixel size.
        aperture_types : tuple of str or None
            Optional aperture type filter, e.g. ``("OH",)`` for an object-hole
            mask only. ``None`` includes every aperture.

        Returns
        -------
        result : np.ndarray
            Return value produced by the function.
        """
        shape = (
            tuple(output_shape)
            if output_shape is not None
            else tuple(self.aperture_shape[-2:])
        )
        pixel_size = output_pixel_size or self.real_space_pixel_size
        if pixel_size <= 0:
            raise ValueError(f"output_pixel_size must be positive, got {pixel_size}")
        supportmask = np.zeros(shape, dtype=np.uint8)
        if self.aperture_method is None:
            return supportmask

        types = self.aperture_config.get("apertures_type", [])
        radii = self.aperture_config.get("apertures_radius", [])
        centers = self.aperture_config.get("apertures_center", [])
        n = len(radii)
        angles = self._aperture_values("apertures_angle", n, 0.0)
        ellipticities = self._aperture_values("apertures_ellipticity", n, 1.0)
        roughnesses = self._aperture_values("apertures_roughness", n, 0.0)
        roughness_modes = self._aperture_values(
            "apertures_roughness_modes", n, (0, 0)
        )
        seeds = self._aperture_values("apertures_seed", n, None)
        top_radius_factors = self._aperture_values(
            "apertures_top_radius_factor", n, 2.0
        )
        lengths = self._aperture_values("apertures_length", n, None)
        center_y0 = shape[0] / 2
        center_x0 = shape[1] / 2

        for (
            type,
            radius,
            center,
            angle,
            ellipticity,
            roughness,
            modes,
            seed,
            top_radius_factor,
            length,
        ) in zip(
            types,
            radii,
            centers,
            angles,
            ellipticities,
            roughnesses,
            roughness_modes,
            seeds,
            top_radius_factors,
            lengths,
        ):
            if aperture_types is not None and type not in aperture_types:
                continue
            center_y = center_y0 + center[0] / pixel_size
            center_x = center_x0 + center[1] / pixel_size
            seed = None if seed is None or int(seed) < 0 else int(seed)
            if type == "SLIT":
                if length is None:
                    raise ValueError("SLIT apertures require apertures_length.")
                width_px = radius * max(1.0, float(top_radius_factor)) / pixel_size
                length_px = length * max(1.0, float(top_radius_factor)) / pixel_size
                if self.use_roi:
                    y_slice, x_slice = structures.Apertures3D._rectangle_aperture_bbox(
                        (1, *shape),
                        (center_y, center_x),
                        width_px,
                        length_px,
                        sigma=None,
                        angle=angle,
                        roughness=roughness,
                    )
                    local_shape = (
                        1,
                        y_slice.stop - y_slice.start,
                        x_slice.stop - x_slice.start,
                    )
                    local_center = (center_y - y_slice.start, center_x - x_slice.start)
                else:
                    y_slice = slice(0, shape[0])
                    x_slice = slice(0, shape[1])
                    local_shape = (1, *shape)
                    local_center = (center_y, center_x)
                inside_hole = structures.Apertures3D._rectangle_aperture_hole_mask(
                    local_shape,
                    local_center,
                    width_px,
                    length_px,
                    sigma=None,
                    angle=angle,
                    roughness=roughness,
                    roughness_modes=tuple(modes),
                    seed=seed,
                )
            else:
                radius_px = radius * max(1.0, float(top_radius_factor)) / pixel_size
                if self.use_roi:
                    y_slice, x_slice = structures.Apertures3D._aperture_bbox(
                        (1, *shape),
                        (center_y, center_x),
                        radius_px,
                        sigma=None,
                        angle=angle,
                        ellipticity=ellipticity,
                        roughness=roughness,
                    )
                    local_shape = (
                        1,
                        y_slice.stop - y_slice.start,
                        x_slice.stop - x_slice.start,
                    )
                    local_center = (center_y - y_slice.start, center_x - x_slice.start)
                else:
                    y_slice = slice(0, shape[0])
                    x_slice = slice(0, shape[1])
                    local_shape = (1, *shape)
                    local_center = (center_y, center_x)
                inside_hole = structures.Apertures3D._aperture_hole_mask(
                    local_shape,
                    local_center,
                    radius_px,
                    sigma=None,
                    angle=angle,
                    ellipticity=ellipticity,
                    roughness=roughness,
                    roughness_modes=tuple(modes),
                    seed=seed,
                )
            supportmask_roi = supportmask[y_slice, x_slice]
            supportmask_roi[inside_hole > 0] = 1
        return supportmask

    def visualize_aperture(self) -> None:
        """Display the depth-averaged aperture mask in pixel and real-space units.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
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
        (e.g. ``distance``, ``fwhm``, and ``alpha_beam=(alpha_y, alpha_x)`` for
        a Gaussian beam).
    """

    XRayConfig: XRayConfig
    shape: tuple[int, int]
    real_space_pixel_size: float
    illumination_function: Literal["gaussian"] | None = "gaussian"
    illumination_config: dict = field(default_factory=dict)

    def _apply_illumination_function(self) -> None:
        """Apply the current illumination function to the existing illumination object.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        if self.illumination_function == "gaussian":
            self.illumination.gauss_beam(**self.illumination_config)
        elif self.illumination_function in ("plane_wave", None):
            self.illumination.plane_wave(self.shape)

    def _incident_photons_per_second(self) -> float:
        """Handle the internal incident photons per second operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : float
            Return value produced by the function.
        """
        return self.XRayConfig.photon_flux

    def _scale_illumination_to_photon_flux(self) -> None:
        """Handle the internal scale illumination to photon flux operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        target_intensity = self._incident_photons_per_second()
        current_intensity = np.sum(np.abs(self.illumination.illumination) ** 2)
        if current_intensity <= 0:
            raise ValueError("Cannot scale illumination with zero total intensity.")
        self.illumination.illumination *= np.sqrt(target_intensity / current_intensity)

    def setup(self) -> None:
        """Compute the spatial wavefield and initialise Jones vectors.

        Builds the beam envelope (Gaussian or plane wave) from the current
        ``XRayConfig``. Expensive — call once. Use ``update_polarization()``
        to switch polarisation state without rebuilding the envelope.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        self.beam_params = self.XRayConfig.setup()
        self.illumination = light_beam.illumination(
            self.beam_params, self.shape, self.real_space_pixel_size
        )
        self._apply_illumination_function()
        self._scale_illumination_to_photon_flux()
        self.illumination.get_illumination_jones()

    def update_polarization(self, pol: str) -> None:
        """Switch polarisation and recompute Jones vectors without rebuilding the wavefield.

        Parameters
        ----------
        pol : {"CR", "CL", "x", "y"}
            New polarisation state.

        Returns
        -------
        None
            The function completes in place.
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

        Returns
        -------
        None
            The function completes in place.
        """
        self.illumination_config = illumination_config
        self._apply_illumination_function()
        self._scale_illumination_to_photon_flux()
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

        Returns
        -------
        None
            The function completes in place.
        """
        self.XRayConfig.energy = energy
        self.beam_params = self.XRayConfig.setup()
        self.illumination.beam_parameters = self.beam_params
        self._apply_illumination_function()
        self._scale_illumination_to_photon_flux()
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

        Returns
        -------
        None
            The function completes in place.
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

    def get_metadata(self, prefix: str = "") -> dict[str, int | float | str | bool]:
        """Return config fields plus derived beam parameters as scalar metadata.

        Parameters
        ----------
        prefix : str
            Input value for ``prefix``.

        Returns
        -------
        result : dict[str, int | float | str | bool]
            Return value produced by the function.
        """
        meta = super().get_metadata(prefix=prefix)
        if hasattr(self, "beam_params"):
            bp = self.beam_params
            bp_prefix = f"{prefix}beam_params/"
            for attr in ("energy", "wavelength", "wavevector", "pol", "photon_flux", "coherence_length"):
                meta[f"{bp_prefix}{attr}"] = getattr(bp, attr)
        return meta


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

    def _illumination_jones_for_shape(self, shape: tuple[int, int]) -> np.ndarray:
        """Evaluate the configured illumination on ``shape`` with original scaling."""
        beam_params = self.IlluminationConfig.beam_params
        pixel_size = self.SampleConfig.real_space_pixel_size

        def build_unscaled(target_shape):
            illum = light_beam.illumination(beam_params, target_shape, pixel_size)
            if self.IlluminationConfig.illumination_function == "gaussian":
                illum.gauss_beam(**self.IlluminationConfig.illumination_config)
            elif self.IlluminationConfig.illumination_function in ("plane_wave", None):
                illum.plane_wave(target_shape)
            else:
                raise ValueError(
                    "Far-field background padding only supports gaussian, "
                    "plane_wave, or None illumination functions."
                )
            return light_beam.scalar_to_jones(illum.illumination, beam_params.pol)

        original_shape = self.IlluminationConfig.shape
        original_unscaled = build_unscaled(original_shape)
        original_scaled = self.IlluminationConfig.illumination.illumination_jones
        unscaled_power = np.sum(np.abs(original_unscaled) ** 2)
        scaled_power = np.sum(np.abs(original_scaled) ** 2)
        if unscaled_power <= 0 or scaled_power <= 0:
            raise ValueError("Cannot build far-field background from zero illumination.")
        scale = np.sqrt(scaled_power / unscaled_power)
        return build_unscaled(shape) * scale

    def _background_exit_jones(self, shape: tuple[int, int]) -> np.ndarray:
        """Build the ROI-consistent background exit wave on an extended grid."""
        background = self._illumination_jones_for_shape(shape)
        eps_stack = self.SampleConfig.sample_structure.final_dielectric_tensor
        thicknesses = self.SampleConfig.sample_structure.layer_thicknesses
        wavelength = self.IlluminationConfig.beam_params.wavelength
        k0 = 2 * np.pi / wavelength
        propagate = bool(self.propagator_config.get("propagate", False))
        jones_apply_zero_order_phase = bool(
            self.propagator_config.get("jones_apply_zero_order_phase", True)
        )

        if hasattr(eps_stack, "base_diagonal"):
            base_diagonal = eps_stack.base_diagonal
        else:
            eps_arr = np.asarray(eps_stack)
            base_diagonal = np.stack(
                (eps_arr[:, 0, 0, 0, 0], eps_arr[:, 0, 0, 1, 1]),
                axis=-1,
            )

        for iz, dz in enumerate(thicknesses):
            phase = -1j * k0 * float(dz)
            background[..., 0] *= np.exp(phase * np.sqrt(base_diagonal[iz, 0]))
            background[..., 1] *= np.exp(phase * np.sqrt(base_diagonal[iz, 1]))
            if iz < len(thicknesses) - 1 and (
                propagate or jones_apply_zero_order_phase
            ):
                background *= np.exp(-1j * k0 * float(dz))
        return background

    def setup(self) -> Jones_propagator.wavefronts:
        """Build the beam propagator object based on the current sample and illumination.

        Returns
        -------
        Jones_propagator.JonesPropagator
            The configured beam propagator object.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """
        if self.propagator_method == "Jones":
            self.wavefront = self._jones_propagation()
        elif self.propagator_method is None:
            return None
        else:
            raise ValueError(f"Unknown propagator method: {self.propagator_method!r}")

    def _jones_propagation(self):
        """Handle the internal jones propagation operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        farfield_oversampling = int(
            self.propagator_config.get("farfield_oversampling", 1)
        )
        farfield_background_jones = None
        if farfield_oversampling > 1:
            ny, nx = self.IlluminationConfig.shape
            farfield_background_jones = self._background_exit_jones(
                (ny * farfield_oversampling, nx * farfield_oversampling)
            )

        wavefront = Jones_propagator.wavefronts(
            beam_parameters=self.IlluminationConfig.beam_params,
            eps_stack=self.SampleConfig.sample_structure.final_dielectric_tensor,
            layer_thicknesses=self.SampleConfig.sample_structure.layer_thicknesses,
            real_space_pixel_size=self.SampleConfig.real_space_pixel_size,
            E_in=self.IlluminationConfig.illumination.illumination_jones,
            aperture_support_regions=getattr(
                self.SampleConfig.sample_structure,
                "aperture_support_regions",
                None,
            ),
            propagate=bool(self.propagator_config.get("propagate", False)),
            jones_apply_zero_order_phase=bool(
                self.propagator_config.get("jones_apply_zero_order_phase", True)
            ),
            propagation_padding_px=int(
                self.propagator_config.get("propagation_padding_px", 0)
            ),
            propagation_padding_mode=str(
                self.propagator_config.get("propagation_padding_mode", "edge")
            ),
            propagation_absorber_width_px=int(
                self.propagator_config.get("propagation_absorber_width_px", 0)
            ),
            propagation_absorber_strength=float(
                self.propagator_config.get("propagation_absorber_strength", 0.0)
            ),
            propagation_absorber_profile=str(
                self.propagator_config.get("propagation_absorber_profile", "cosine")
            ),
            multislice_propagation_roi=bool(
                self.propagator_config.get("multislice_propagation_roi", False)
            ),
            multislice_propagation_roi_padding_px=int(
                self.propagator_config.get(
                    "multislice_propagation_roi_padding_px", 0
                )
            ),
            multislice_propagation_roi_merge_overlaps=bool(
                self.propagator_config.get(
                    "multislice_propagation_roi_merge_overlaps", True
                )
            ),
            farfield_oversampling=farfield_oversampling,
            farfield_background_jones=farfield_background_jones,
        )
        return wavefront

    def return_wavefront(self) -> Jones_propagator.wavefronts:
        """Return the configured wavefront object.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : Jones_propagator.wavefronts
            Return value produced by the function.
        """
        return self.wavefront

    def calculate_scalar_wavefield(self) -> np.ndarray:
        """Run the calculate scalar wavefield operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : np.ndarray
            Return value produced by the function.
        """
        amp = (
            np.abs(self.wavefront.exit_wave[..., 0]) ** 2
            + np.abs(self.wavefront.exit_wave[..., 1]) ** 2
        ) / np.sqrt(2)
        phase = np.angle(self.wavefront.exit_wave[..., 0])

        self.exit_wavefield = amp * np.exp(1j * phase)

    def return_scalar_wavefield(self) -> np.ndarray:
        """Return the scalar exit wavefield after sample propagation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : np.ndarray
            Return value produced by the function.
        """
        if not hasattr(self, "exit_wavefield"):
            self.calculate_scalar_wavefield()
        return self.exit_wavefield

    def visualize_exit_wavefront(self) -> None:
        """Display the intensity and phase of the exit wavefront.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
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

    def get_metadata(self, prefix: str = "") -> dict[str, int | float | str | bool]:
        """Return own fields plus SampleConfig and IlluminationConfig metadata.

        Parameters
        ----------
        prefix : str
            Input value for ``prefix``.

        Returns
        -------
        result : dict[str, int | float | str | bool]
            Return value produced by the function.
        """
        meta = super().get_metadata(prefix=prefix)
        meta.update(self.SampleConfig.get_metadata(prefix=f"{prefix}sample/"))
        meta.update(self.IlluminationConfig.get_metadata(prefix=f"{prefix}illumination/"))
        return meta


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
    detected_holograms_no_beamstop: dict = field(default_factory=dict)
    exit_waves: dict = field(default_factory=dict)
    detector_layout: detector.detector_layout | None = None
    sample_x: NDArray[np.float64] | None = None
    sample_y: NDArray[np.float64] | None = None

    _VALID_HELICITIES: tuple = field(
        default=("CR", "CL", "x", "y"), init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Handle the internal post init operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        for store_name, store in (
            ("ideal_holograms", self.ideal_holograms),
            ("detected_holograms", self.detected_holograms),
            (
                "detected_holograms_no_beamstop",
                self.detected_holograms_no_beamstop,
            ),
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

    def _get_store(
        self,
        source: Literal["ideal", "detected", "detected_no_beamstop", "exit_wave"],
    ) -> dict:
        """Handle the internal get store operation.

        Parameters
        ----------
        source : Literal["ideal", "detected", "detected_no_beamstop", "exit_wave"]
            Input value for ``source``.

        Returns
        -------
        result : dict
            Return value produced by the function.
        """
        if source == "ideal":
            return self.ideal_holograms
        if source == "detected":
            return self.detected_holograms
        if source == "detected_no_beamstop":
            return self.detected_holograms_no_beamstop
        if source == "exit_wave":
            return self.exit_waves
        raise ValueError(
            f"Unknown source {source!r}. Must be 'ideal', 'detected', "
            "'detected_no_beamstop', or 'exit_wave'."
        )

    @property
    def helicities(self) -> list[str]:
        """Helicity keys present in the ideal hologram dict.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : list[str]
            Return value produced by the function.
        """
        return list(self.ideal_holograms.keys())

    def add_ideal_hologram(self, helicity: str, hologram: np.ndarray) -> None:
        """Add or replace an ideal hologram for *helicity*.

        *hologram* may be 2-D ``(Ny, Nx)`` or 3-D ``(N_frames, Ny, Nx)``.

        Parameters
        ----------
        helicity : str
            Input value for ``helicity``.
        hologram : np.ndarray
            Input value for ``hologram``.

        Returns
        -------
        None
            The function completes in place.
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

        Parameters
        ----------
        helicity : str
            Input value for ``helicity``.
        hologram : np.ndarray
            Input value for ``hologram``.

        Returns
        -------
        None
            The function completes in place.
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
        """Concatenate *data* arrays into *store* along axis 0.

        Parameters
        ----------
        store : dict
            Input value for ``store``.
        data : dict
            Input value for ``data``.

        Returns
        -------
        None
            The function completes in place.
        """
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
        source: Literal["ideal", "detected", "detected_no_beamstop"] = "detected",
    ) -> None:
        """Stack new holograms onto the existing store for each helicity.

        Parameters
        ----------
        holograms : dict[str, ndarray]
            Mapping of helicity → array. Each array may be 2-D ``(Ny, Nx)``
            or 3-D ``(N_frames, Ny, Nx)``.
        source : {"ideal", "detected", "detected_no_beamstop"}
            Which hologram store to append to.

        Returns
        -------
        None
            The function completes in place.
        """
        self._stack_into(self._get_store(source), holograms)

    def add_exit_waves(self, exit_waves: dict) -> None:
        """Stack new scalar exit wavefields onto the exit-wave store.

        Parameters
        ----------
        exit_waves : dict[str, ndarray]
            Mapping of helicity → complex array. Each array may be 2-D
            ``(Ny, Nx)`` or 3-D ``(N_frames, Ny, Nx)``.

        Returns
        -------
        None
            The function completes in place.
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

        Returns
        -------
        result : np.ndarray
            Return value produced by the function.
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

        Returns
        -------
        result : np.ndarray
            Return value produced by the function.
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
        """Run the compute differences operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        for source in ["ideal", "detected", "exit_wave"]:
            self.difference(source)

    def compute_sums(self):
        """Run the compute sums operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        for source in ["ideal", "detected", "exit_wave"]:
            self.sum(source)

    def compute_reconstructions(self):
        """Run the compute reconstructions operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        for source in ["ideal", "detected", "exit_wave"]:
            store = self._get_store(source)
            for helicity in list(store.keys()):
                self.reconstruct(source, helicity)

    @staticmethod
    def _fth_reconstruct(holo: np.ndarray) -> np.ndarray:
        """Handle the internal fth reconstruct operation.

        Parameters
        ----------
        holo : np.ndarray
            Input value for ``holo``.

        Returns
        -------
        result : np.ndarray
            Return value produced by the function.
        """
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

        Returns
        -------
        None
            The function completes in place.
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

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
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

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
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
        sources: list[
            Literal["exit_wave", "ideal", "detected", "detected_no_beamstop"]
        ]
        | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return exit waves, ideal holograms, and detected holograms grouped by helicity.

        Parameters
        ----------
        helicities : list of str or None
            Helicity keys to include, e.g. ``["CR", "CL"]``.
            If ``None`` (default), all available helicities are returned.
        sources : list of {"exit_wave", "ideal", "detected", "detected_no_beamstop"} or None
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
            "detected_no_beamstop": self.detected_holograms_no_beamstop,
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
