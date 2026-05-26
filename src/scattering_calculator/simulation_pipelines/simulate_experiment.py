import numpy as np

from scattering_calculator.experimental_conditions import detector, light_beam
from scattering_calculator.sample_generator import pattern_generator
from scattering_calculator.sample_generator import structures
from scattering_calculator.beam_propagator import Jones_propagator


class SetupSimulationExperiment:
    def __init__(
        self,
        xray_config,
        simulation_config,
        front_aperture_config,
        illumination_config,
        sample_config,
        magnetic_pattern_config,
        detector_config,
        beamstop_config,
    ):
        self.xray_config = xray_config
        self.simulation_config = simulation_config
        self.front_aperture_config = front_aperture_config
        self.illumination_config = illumination_config
        self.sample_config = sample_config
        self.magnetic_pattern_config = magnetic_pattern_config
        self.detector_config = detector_config
        self.beamstop_config = beamstop_config

    def setup(self):
        self._setup_beam_params()
        self._setup_detector()
        self._setup_beamstop()
        self._setup_sample_geometry()
        self._setup_sample_structure()
        self._setup_front_aperture()
        self._setup_magnetization()

    def run(self):
        self.sample.calculate_final_dielectric_tensor()
        self.holos = [None, None]
        for ii, pol in enumerate(["CR", "CL"]):
            self.beam_params.pol = pol
            illumination = light_beam.illumination(
                self.beam_params,
                self.sample_shape[1:],
                self.real_space_pixel_size,
            )
            illumination.gauss_beam(
                self.illumination_config.illumination_center,
                self.illumination_config.illumination_focus_distance,
                self.illumination_config.illumination_fwhm,
            )
            illumination.get_illumination_jones()

            wavefront = Jones_propagator.wavefronts(
                beam_parameters=self.beam_params,
                eps_stack=self.sample.final_dielectric_tensor,
                layer_thicknesses=self.sample.layer_thicknesses,
                real_space_pixel_size=self.sample.real_space_pixel_size,
                E_in=illumination.illumination_jones,
                propagate=False,
            )

            hologram_exp = detector.detector_hologram(
                self.exp_detector,
                wavefront.hologram,
                self.beam_params,
                self.sample.real_space_pixel_size,
                self.beamstop,
                artifacts_config=getattr(self.detector_config, "artifacts_config", None),
                measurement_config=getattr(
                    self.detector_config, "measurement_config", None
                ),
                detector_params=getattr(self.detector_config, "detector_params", None),
                coherence_length=self.beam_params.coherence_length,
            )
            hologram_exp.gnomonic_projection()
            hologram_exp.add_noise()
            self.holos[ii] = hologram_exp.hologram_exp.copy()

    def _setup_beam_params(self):
        self.beam_params = light_beam.beam_parameters(
            getattr(
                self.xray_config,
                "x_ray_energy",
                getattr(self.xray_config, "energy", None),
            ),
            getattr(self.xray_config, "pol", "CR"),
            getattr(
                self.xray_config,
                "x_ray_photon_flux",
                getattr(self.xray_config, "photon_flux", 1e12),
            ),
            getattr(self.xray_config, "coherence_length", (10e-6, 10e-6)),
        )
        self.beam_params.calc_wavevector()

    def _setup_detector(self):
        self.exp_detector = detector.detector_layout(
            pixel_size=self.detector_config.detector_pixel_size,
            detector_shape=self.detector_config.detector_pixel_shape,
            distance_sample_detector=self.detector_config.detector_distance,
            detector_center=self.detector_config.detector_center,
        )
        self.exp_detector.calc_real_space_coordinates()
        self.exp_detector.calc_q_space_coordinates(self.beam_params)

    def _setup_beamstop(self):
        if hasattr(self.beamstop_config, "setup"):
            self.beamstop = self.beamstop_config.setup(self.exp_detector)
            self.exp_detector.assign_beamstop(self.beamstop.return_beamstop())
            return

        self.beamstop = detector.beamstop(
            self.exp_detector, self.beamstop_config.beamstop_distance
        )
        beamstop_kwargs = dict(getattr(self.beamstop_config, "bs_config", {}))
        if not beamstop_kwargs and hasattr(self.beamstop_config, "beamstop_radius"):
            beamstop_kwargs["radius"] = self.beamstop_config.beamstop_radius
        self.beamstop.create_circle_beamstop(
            self.beamstop_config.beamstop_center,
            use_real_space_coordinates=True,
            **beamstop_kwargs,
        )
        self.exp_detector.assign_beamstop(self.beamstop.return_beamstop())

    def _setup_sample_geometry(self):
        res_from_detector = np.pi / np.maximum(
            np.amax(np.abs(self.exp_detector.detqx)),
            np.amax(np.abs(self.exp_detector.detqy)),
        )
        self.sample_shape = np.array([
            0,
            int(2 * self.detector_config.detector_pixel_shape[0]),
            int(2 * self.detector_config.detector_pixel_shape[0]),
        ])
        self.real_space_pixel_size = res_from_detector / 4

    def _setup_sample_structure(self):
        stack = structures.parse_recipe(
            self.sample_config.recipe,
            sample_name=self.sample_config.sample_name,
            comments=self.sample_config.comments,
        )
        material_params = structures.material_params(
            materials=set(
                material
                for layer in stack.layers
                for material, _ in (
                    layer.components or ((layer.material, layer.thickness),)
                )
            ),
            x_ray_energy=self.xray_config.x_ray_energy,
        )
        self.sample = structures.Structure(
            name=self.sample_config.name,
            material_params=material_params,
            sample_shape=self.sample_shape,
            real_space_pixel_size=self.real_space_pixel_size,
        )
        for layer in stack.layers:
            if layer.is_composite:
                self.sample.add_effective_layer(layer.material, layer.components)
            else:
                self.sample.add_layer(layer.material, thickness=layer.thickness)

    def _setup_front_aperture(self):
        self.front_aperture = structures.Apertures3D(
            self.sample_shape,
            self.real_space_pixel_size,
            layer_thicknesses=self.sample.layer_thicknesses,
        )
        for i, aperture_type in enumerate(self.front_aperture_config.apertures_type):
            if aperture_type == "RH":
                depth = (np.sum(self.sample.layer_thicknesses),)
            else:
                depth = np.sum(np.array(
                    self.sample.layer_thicknesses[
                        : self.sample.layer_names.index("SiN")
                    ]
                ))
            self.front_aperture.create_circle_aperture(
                center=self.front_aperture_config.apertures_centers[i],
                depth=depth,
                radius=self.front_aperture_config.apertures_radius[i],
                use_real_space_coordinates=True,
                sigma=self.front_aperture_config.apertures_sigma[i],
            )
        self.sample.mask = self.front_aperture.aperture_design

    def _setup_magnetization(self):
        magnetic_pattern, _ = pattern_generator.create_wavy_stripe_pattern(
            self.sample.sample_shape[1:],
            self.magnetic_pattern_config.stripe_width / self.real_space_pixel_size,
            self.magnetic_pattern_config.angle_stripes,
            self.magnetic_pattern_config.sigma / self.real_space_pixel_size,
            True,
            self.real_space_pixel_size,
            waviness_amplitude=self.magnetic_pattern_config.wave_amplitudes / self.real_space_pixel_size,
            waviness_scale=self.magnetic_pattern_config.wave_scale / self.real_space_pixel_size,
            seed=self.magnetic_pattern_config.seed,
        )
        self.sample.magnetization = pattern_generator.map_magnetization_to_3d(
            0 * magnetic_pattern,
            np.sqrt(1 - np.abs(magnetic_pattern) ** 2),
            magnetic_pattern,
            nr_repeats=self.sample.sample_shape[0],
        )
