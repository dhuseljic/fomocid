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
    energy: float  # eV
    photon_flux: float  # photons/s
    polarization: Literal["CR", "CL", "x", "y"] = "CR"
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
        if self.polarization not in ["CR", "CL", "x", "y"]:
            raise ValueError(
                f"Polarisation must be in CR, CL, x or, got {self.polarization}"
            )

    def setup(self) -> light_beam.beam_parameters:
        self.beam_params = light_beam.beam_parameters(
            self.energy, self.photon_flux, self.polarization, self.coherence_length
        )
        self.beam_params.calc_wavevector()
        self.wavevector = self.beam_params.wavevector
        return self.beam_params


@dataclass
class BeamstopConfig(_ConfigMixin):
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
        bs = detector.beamstop(
            detector_config=detector_layout,
            distance_detector_beamstop=self.bs_detector_distance,
        )
        if self.bs_method is None:
            return bs.create_empty_beamstop()
        if self.bs_method == "circular":
            bs.create_circle_beamstop(center=self.bs_center, **self.bs_config)
        return bs


@dataclass
class DetectorConfig(_ConfigMixin):
    shape: tuple[int, int] = (256, 256)  # px
    pixel_size: float = 55e-6  # m/px
    sample_to_detector_distance: float = 0.1  # m
    detector_center: tuple[int, int] = (0, 0)  # px
    detector_efficiency: float = 1.0  # 0–1
    detector_noise_rms: float = 0.0  # counts
    artifacts_method: str | None = None
    artifacts_config: dict = field(default_factory=dict)
    beamstop: BeamstopConfig | None = None

    def __post_init__(self) -> None:
        if any(s <= 0 for s in self.shape):
            raise ValueError(f"shape dimensions must be positive, got {self.shape}")
        if self.pixel_size <= 0:
            raise ValueError(f"pixel_size must be positive, got {self.pixel_size}")
        if self.sample_to_detector_distance <= 0:
            raise ValueError(
                f"sample_to_detector_distance must be positive, got {self.sample_to_detector_distance}"
            )
        if not (0.0 <= self.detector_efficiency <= 1.0):
            raise ValueError(
                f"detector_efficiency must be in [0, 1], got {self.detector_efficiency}"
            )
        if self.detector_noise_rms < 0:
            raise ValueError(
                f"detector_noise_rms must be non-negative, got {self.detector_noise_rms}"
            )

    def setup(self):
        self.detector_layout = detector.detector_layout(
            pixel_size=self.pixel_size,
            detector_shape=self.shape,
            distance_sample_detector=self.sample_to_detector_distance,
            detector_center=self.detector_center,
        )
        if self.beamstop is not None:
            bs = self.beamstop.setup(self.detector_layout)
            self.detector_layout.assign_beamstop(bs.beamstop)
        return self.detector_layout

    def calc_realspace_resolution(self, beam_parameters: light_beam.beam_parameters):
        self.detector_layout.calc_q_space_coordinates(beam_parameters)
        self.detector_layout.calc_resolution_from_detector()

        return self.detector_layout.real_space_resolution


@dataclass
class SimulationConfig(_ConfigMixin):
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

    def setup(self) -> dict:
        y = (np.arange(self.shape[0]) - self.shape[0] / 2) * self.real_space_pixel_size
        x = (np.arange(self.shape[1]) - self.shape[1] / 2) * self.real_space_pixel_size
        self.xgrid, self.ygrid = np.meshgrid(x, y)


@dataclass
class SampleConfig(_ConfigMixin):
    recipe: str = ("Recipe",)
    sample_shape: tuple[int, int, int] | None = (None,)
    real_space_pixel_size: float | None = (None,)
    xray_config: XRayConfig | None = (None,)
    sample_name: str | None = (None,)
    comments: str | None = (None,)
    other_config: dict = field(default_factory=dict)

    def setup(self) -> structures.MultilayerRecipe:
        # Define stack
        self.multilayer_recipe = structures.parse_recipe(
            self.recipe,
            sample_name=self.sample_name,
            comments=self.comments,
        )

        # Load material parameters
        self.material_params = structures.material_params(
            materials=set(
                [element.material for element in self.multilayer_recipe.layers]
            ),
            x_ray_energy=self.xray_config.energy,
        )

        # Create sample structure
        self.sample_structure = structures.Structure(
            name=self.sample_name,
            material_params=self.material_params,
            sample_shape=self.sample_shape,
            real_space_pixel_size=self.real_space_pixel_size,
        )
        for layer in self.multilayer_recipe.layers:
            self.sample_structure.add_layer(layer.material, thickness=layer.thickness)

    def assign_magnetic_pattern(self, magnetic_pattern: np.ndarray):
        self.magnetic_pattern = magnetic_pattern
        self.magnetization = pattern_generator.map_magnetization_to_3d(
            0 * magnetic_pattern,
            np.sqrt(1 - np.abs(magnetic_pattern) ** 2),
            magnetic_pattern,
            nr_repeats=self.sample_shape[0],
        )


@dataclass
class MagneticPatternConfig(_ConfigMixin):
    pattern_type_method: Literal["skyrmion_pattern", "wavy_stripe_pattern"] = (
        "skyrmion_pattern"
    )
    shape: tuple[int, int] | None = None
    real_space_pixel_size: float | None = None
    pattern_config: dict = field(default_factory=dict)
    pattern_config_length: dict = field(default_factory=dict)

    def setup(self):
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

    def create_pattern(self):
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

    def plot_pattern(self):
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
    aperture_method: Literal["circular", "rectangular"] | None = "circular"
    aperture_thickness: float = 0.01  # m
    aperture_center: tuple[int, int] = (0, 0)  # px
    aperture_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.aperture_thickness <= 0:
            raise ValueError(
                f"aperture_thickness must be positive, got {self.aperture_thickness}"
            )

    def setup(
        self, shape: tuple[int, int], real_space_pixel_size: float
    ) -> structures.Apertures2D:
        aperture = structures.Apertures2D(shape, real_space_pixel_size)
        if self.aperture_method == "circular":
            aperture.create_circle_aperture(
                center=self.aperture_center,
                **self.aperture_config,
            )
        return aperture


@dataclass
class IlluminationConfig(_ConfigMixin):
    illumination_function: Literal["gaussian"] | None = "gaussian"
    illumination_center: tuple[int, int] = (0, 0)  # px
    illumination_config: dict = field(default_factory=dict)

    def setup(
        self,
        beam_params: light_beam.beam_parameters,
        shape: tuple[int, int],
        real_space_pixel_size: float,
    ) -> light_beam.illumination:
        illum = light_beam.illumination(beam_params, shape, real_space_pixel_size)

        if self.illumination_function == "gaussian":
            illum.gauss_beam(
                center=self.illumination_center, **self.illumination_config
            )
        elif self.illumination_function is None:
            illum.plane_wave(shape)
        return illum
