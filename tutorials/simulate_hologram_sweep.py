"""Generate parameter sweeps of simulated FTH holograms."""

# %%
# Import general libraries
import os
import numpy as np
import h5py
from scattering_calculator.utils import physics
import ast

# Imports from our own codebase
from fomocid import DATA_ROOT
from scattering_calculator.simulation_pipelines import (
    Uniform,
    Choice,
    FrontApertureConfig,
)
from scattering_calculator.simulation_pipelines.pipelines import (
    HologramPipeline,
    HologramPipelineConfig,
    HologramPipelineRanges,
)
from scattering_calculator.sample_generator import structures


#################################################################
#### HOW MANY SIMULATIONS TO RUN? ####
#################################################################
nr_simulations = 2  # increase to e.g. 1000 for a full training dataset

# --- Material stack ---
recipe = "[Au(140)Cr(60)]x5/SiN(200)/Pt(20)Al(20)Co(20)"

# simulation sampling options; these can be overridden in the ranges below to create mixed sampling
oversampling=2 # sampling of the sample relative to the dector-based sampling; e.g. oversampling=2 means the sample grid has 2x finer pixel size than the detector-projected pixel size in the sample plane; this is separate from farfield_oversampling, which controls the hologram sampling relative to the detector
farfield_oversampling = 2  # >1 extends the exit wave with the physical background before the far-field FFT
detector_pixel_footprint_samples = 2  # sub-samples per detector-pixel axis when footprint averaging is enabled

# %%
# ===================
# OUTPUT PATH
# ===================
output_folder = DATA_ROOT / "Data" / "hologram_sweep"
output_path = output_folder / "simulation_sweep.h5"


# Pipeline settings that are fixed across all runs in the sweep. These can be overridden in the ranges below to create mixed sampling, but any field not mentioned in the ranges will always use these values.
pipeline_random_seed = 0  # set to None for non-reproducible random sweeps
use_roi = True # set True to use a region of interest around the sample for the whole pipeline, which can greatly speed up simulations with large free-space regions; set False to use the full grid, which can improve accuracy for large beamstop distances or very wide beamstops but uses more memory and computation time
dielectric_tensor_use_roi = True # set True to only compute the dielectric tensor in a region of interest around the sample, which can greatly speed up simulations with large free-space regions; set False to compute the full dense tensor stack, which can improve accuracy for large beamstop distances or very wide beamstops but uses more memory and computation time
dielectric_tensor_compact = True  # avoid allocating the full dense tensor stack
propagate = True  # set True for multislice free-space propagation between layers
multislice_propagation_roi = True  # if True, propagate aperture ROI crops and add local corrections to the plane-wave baseline
multislice_propagation_roi_padding_px = 64  # enlarges ROI boxes around apertures and gives the local correction taper room to fade smoothly
multislice_propagation_roi_merge_overlaps = True  # merge overlapping padded ROI crops before propagation; physical or padded overlaps are always merged even if this is False
propagation_padding_px = 128  # 0 disables padded free-space propagation
propagation_padding_mode = "edge"  # "edge", "reflect", "symmetric", or "constant"
propagation_absorber_width_px = 64  # 0 disables edge absorption
propagation_absorber_strength = 6.0  # larger values damp padded-edge wraparound more
propagation_absorber_profile = "cosine"  # "cosine", "smoothstep", "quadratic", or "linear"
save_detected_hologram_without_beamstop = True # set True to save an extra detected hologram without the beamstop shadow (for supervised learning or beamstop ablation studies)
use_detector_pixel_footprint = True  # True = average the ideal hologram over each finite detector pixel


os.makedirs(output_folder, exist_ok=True)

# %%
# ===================
# FIXED PARAMETERS
# ===================
# These are the baseline values used whenever a parameter is NOT swept.
# Any field here can be moved into HologramPipelineRanges to vary it.

# --- X-ray source ---
x_ray_energy = 787.9  # eV  (Co L-edge)
x_ray_photon_flux = 1e12  # photons/pulse
coherence_length = (100e-6, 100e-6)  # m, (y, x)

# --- Detector ---
detector_pixel_size = 20e-6  # m/px
detector_pixel_shape = (1300, 1300)
detector_distance = 0.02  # m
detector_center = (
    detector_pixel_shape[0] / 2,
    detector_pixel_shape[1] / 2,
)  # px, (y, x) direct-beam position on the detector
detector_center_jitter_fraction = 0.05  # sample detector_center within +/-5% of each detector axis
beamstop_center_jitter_radius_fraction = 0.5  # sample beamstop offset within +/-0.5 projected beamstop radii


detector_params = {
    "readout_noise_average": 50,
    # Canonical readout-noise width. The legacy name "noise_rms" is only an alias.
    "readout_noise_sigma": 3,
    "detector_threshold": 64.3e3,
    # Canonical conversion between detector counts and photon events.
    "counts_per_photon": 100,
    "quantum_efficiency": 1.0,
}
# Acquisition timing/frame settings are separate from detector response.
measurement_config = {
    "number_frames": 1,
    "max_counts_per_image": 64.3e3,
    "exposure_time": 1e-2,
}
# Photon-event shape controls only; counts_per_photon belongs in detector_params.
artifacts_config = {
    "sigma_photon": 0.75,
    "photon_n_classes": 1,
    "photon_n_variants": 30,
    "photon_kernel_size": 9,
    "photon_irregularity": 2.0,
    "regenerate_photon_kernels": True,
}

# --- Beamstop ---
beamstop_method = "circular"
beamstop_distance = 0.001  # m
beamstop_config = {
    "radius": 0.2e-3,
    "angle": 0,
    "sigma": 20e-6,
    "ellipticity": (0.8, 1.2),
    "roughness": 0.05,
    "roughness_modes": (3, 9),
    "wire_width": 0.05e-3,
    "wire_bend": 0.1e-3,
    "antialias": 2,
    "seed": None,
}



# --- Magnetic domain pattern ---
pattern_type = "binary_labyrinth_pattern"  # "wavy_stripe_pattern", "binary_labyrinth_pattern", "disordered_skyrmion_lattice_pattern", "saturated_pattern", or "image_pattern"
stripe_width = 300e-9  # m
sigma = 30e-9  # m
angle_stripes = np.pi / 4
waviness_amplitude = 0e-9  # ms
waviness_scale = 0e-9  # m
experimental_pattern_path = None  # e.g. DATA_ROOT / "Data" / "reconstruction_domains.png"
experimental_pattern_pixel_size = None  # real-space pixel size of that reconstruction, in m
experimental_pattern_threshold = 0.5
experimental_pattern_invert = False
experimental_pattern_pad_mode = "edge"
saturated_config = {
    "saturation": 1,
}
skyrmion_lattice_config = {
    "skyrmion_density": 0.25,
    "diameter_spread": 0.10 * stripe_width,
    "ellipticity": (0.75, 1.35),
    "roughness": 0.05,
    "roughness_modes": (3, 9),
}
labyrinth_config = {
    "batch": 1,
    "H": 100,
    "W": 100,
    "n_steps": 50,
    "region": "custom",
    "use_gpu": False,
    "k0": 1.0,
    "eps": 0.0,
    "noise_amp": 0.0,
    "domain_conversion": "soft",
    "softness": 1.0,
    "auto_size": True,
    "crop_margin": None,
}
pattern_config = {
    "stripe_width": stripe_width,
    "sigma": sigma,
}
if pattern_type == "wavy_stripe_pattern":
    pattern_config.update(
        {
            "angle_stripes": angle_stripes,
            "waviness_amplitude": waviness_amplitude,
            "waviness_scale": waviness_scale,
        }
    )
elif pattern_type == "binary_labyrinth_pattern":
    pattern_config.update(labyrinth_config)
elif pattern_type == "disordered_skyrmion_lattice_pattern":
    pattern_config.update(skyrmion_lattice_config)
elif pattern_type == "saturated_pattern":
    pattern_config.update(saturated_config)
elif pattern_type == "image_pattern":
    if experimental_pattern_path is None or experimental_pattern_pixel_size is None:
        raise ValueError(
            "image_pattern requires experimental_pattern_path and "
            "experimental_pattern_pixel_size."
        )
    pattern_config.update(
        {
            "image_path": str(experimental_pattern_path),
            "image_pixel_size": experimental_pattern_pixel_size,
            "threshold": experimental_pattern_threshold,
            "invert": experimental_pattern_invert,
            "pad_mode": experimental_pattern_pad_mode,
        }
    )
else:
    raise ValueError(f"Unknown pattern_type: {pattern_type}")

# --- FTH holography mask ---
aperture_types = ["OH", "RH", "RH"]
aperture_radii = [60e-9, 6e-9, 4e-9]  # m
aperture_roughness_amplitude = 20e-9  # m, target boundary fluctuation scale
aperture_roughness_period = 10e-9  # m, target boundary fluctuation period
aperture_centers = [(0, 0), (0.2e-6, -0.15e-6), (0.15e-6, 0.15e-6)]  # m (y, x)
aperture_sigmas = [1e-9, 2e-9, 2e-9]  # m, continuous edge transition widths
aperture_angles = [0.0, 0.0, 0.0]  # rad
aperture_ellipticities = [1.0, 1.0, 1.0]  # y/x axis ratio
max_oh_radius_in_texture_widths = 8.0  # avoids tiny domains inside pathologically huge OHs
max_magnetic_pattern_roi_pixels = 768  # approximate cap for the OH-local magnetic texture ROI


def aperture_roughness_from_length(
    radius,
    amplitude=20e-9,
    period=10e-9,
    max_relative_amplitude=0.25,
):
    """Convert physical roughness amplitude/period to relative Fourier settings.

    Parameters
    ----------
    radius : Any
        Input value for ``radius``.
    amplitude : Any
        Input value for ``amplitude``.
    period : Any
        Input value for ``period``.
    max_relative_amplitude : Any
        Input value for ``max_relative_amplitude``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    relative_amplitude = min(
        float(max_relative_amplitude),
        float(amplitude) / float(radius),
    )
    center_mode = max(1, int(round(2.0 * np.pi * float(radius) / float(period))))
    return relative_amplitude, (max(1, center_mode - 2), center_mode + 2)


aperture_roughnesses, aperture_roughness_modes = zip(
    *[
        aperture_roughness_from_length(
            radius,
            amplitude=aperture_roughness_amplitude,
            period=aperture_roughness_period,
        )
        for radius in aperture_radii
    ]
)
aperture_roughnesses = list(aperture_roughnesses)
aperture_roughness_modes = list(aperture_roughness_modes)
aperture_seeds = [-1, -1, -1]
aperture_top_radius_factors = [2.0, 2.0, 2.0]

# --- Illumination ---
illumination_function = "gaussian"
illumination_center = (0.0, 0.0)  # m
illumination_focus_distance = 1e-3  # m
illumination_fwhm = 10.5e-6  # ms

# %%
# ===================
# PIPELINE CONFIG
# ===================
config = HologramPipelineConfig(
    # Sample material stack
    recipe=recipe,
    sample_name="Co_Pt_multilayer",
    # X-ray source
    xray_energy=x_ray_energy,
    xray_photon_flux=x_ray_photon_flux,
    xray_coherence_length=coherence_length,
    # Detector geometry
    detector_shape=detector_pixel_shape,
    detector_pixel_size=detector_pixel_size,
    detector_distance=detector_distance,
    detector_center=detector_center,
    detector_params=detector_params,
    measurement_config=measurement_config,
    artifacts_config=artifacts_config,
    use_detector_pixel_footprint=use_detector_pixel_footprint,
    detector_pixel_footprint_samples=detector_pixel_footprint_samples,
    # Beamstop
    beamstop_method=beamstop_method,
    beamstop_distance=beamstop_distance,
    beamstop_config=beamstop_config,
    save_detected_hologram_without_beamstop=save_detected_hologram_without_beamstop,
    # FTH holography mask
    aperture_method="FTH_circular",
    aperture_types=aperture_types,
    aperture_radii=aperture_radii,
    aperture_centers=aperture_centers,
    aperture_sigmas=aperture_sigmas,
    aperture_angles=aperture_angles,
    aperture_ellipticities=aperture_ellipticities,
    aperture_roughnesses=aperture_roughnesses,
    aperture_roughness_modes=aperture_roughness_modes,
    aperture_seeds=aperture_seeds,
    aperture_top_radius_factors=aperture_top_radius_factors,
    # Illumination
    illumination_function=illumination_function,
    illumination_center=illumination_center,
    illumination_focus_distance=illumination_focus_distance,
    illumination_fwhm=illumination_fwhm,
    # Magnetic domain pattern
    pattern_type=pattern_type,
    pattern_config=pattern_config,
    use_roi=use_roi,
    magnetic_pattern_use_roi=True,
    dielectric_tensor_use_roi=dielectric_tensor_use_roi,
    dielectric_tensor_compact=dielectric_tensor_compact,
    propagate=propagate,
    multislice_propagation_roi=multislice_propagation_roi,
    multislice_propagation_roi_padding_px=multislice_propagation_roi_padding_px,
    multislice_propagation_roi_merge_overlaps=multislice_propagation_roi_merge_overlaps,
    propagation_padding_px=propagation_padding_px,
    propagation_padding_mode=propagation_padding_mode,
    propagation_absorber_width_px=propagation_absorber_width_px,
    propagation_absorber_strength=propagation_absorber_strength,
    propagation_absorber_profile=propagation_absorber_profile,
    farfield_oversampling=farfield_oversampling,
    # Simulation grid: sample_shape = oversampling * detector_shape
    oversampling=oversampling,
    random_seed=pipeline_random_seed,
)

# %%
# ===================
# PARAMETER RANGES
# ===================
# Define which parameters vary across runs and how they are sampled.
#
# Uniform(low, high)  — draw uniformly from [low, high]
# Choice((a, b, ...)) — pick uniformly from a discrete set
# None                — use the fixed value from HologramPipelineConfig

def random_pattern_config(params):
    """Generate interdependent magnetic-pattern parameters in physical units.

    Parameters
    ----------
    params : Any
        Input value for ``params``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    stripe_width = Uniform(30e-9, 500e-9).sample()
    config = {
        "stripe_width": stripe_width,
        "sigma": Uniform(np.minimum(3e-9,0.01*stripe_width), np.maximum(3e-9,0.12 * stripe_width)).sample(),
    }
    if params["pattern_type"] == "binary_labyrinth_pattern":
        config.update(labyrinth_config)
        return config
    if params["pattern_type"] == "disordered_skyrmion_lattice_pattern":
        ellipticity_delta = Uniform(0.1, 0.5).sample()
        config.update(
            {
                "skyrmion_density": Uniform(0.1, 0.45).sample(),
                "diameter_spread": Uniform(0.01 * stripe_width, 0.10 * stripe_width).sample(),
                "ellipticity": (1.0 - ellipticity_delta, 1.0 + ellipticity_delta),
                "roughness": Uniform(0.01, 0.08).sample(),
                "roughness_modes": (3, 9),
            }
        )
        return config
    if params["pattern_type"] == "saturated_pattern":
        config.update({"saturation": Choice((-1, 1)).sample()})
        return config

    waviness_amplitude = Uniform(0.0, 2.0 * stripe_width).sample()
    min_waviness_scale = max(4.0 * stripe_width, 50e-9)
    max_waviness_scale = 5.0 * stripe_width
    waviness_scale = Uniform(min_waviness_scale, max_waviness_scale).sample()
    config.update(
        {
            "angle_stripes": Uniform(0.0, np.pi).sample(),
            "waviness_amplitude": waviness_amplitude,
            "waviness_scale": waviness_scale,
        }
    )
    return config


def detector_distance_range(params):
    """Choose a detector-distance range from sampled energy and stripe width.

    Parameters
    ----------
    params : Any
        Input value for ``params``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    # This function is called once per simulated sample after x-ray energy,
    # detector pixel size, detector shape, and pattern_config have
    # already been sampled. It returns either a fixed detector distance or a
    # Uniform range that the pipeline samples immediately.
    wavelength = physics.photon_energy_wavelength(params["energy"])
    stripe_width = params["pattern_config"]["stripe_width"]
    detector_width = params["detector_shape"][0] * params["detector_pixel_size"]

    # Upper bound: keep the detector real-space resolution smaller than the
    # sampled texture period. Very long distances are capped at 0.75 m.
    max_resolution = stripe_width
    maximum_detector_distance = detector_width / (
        2 * np.tan(wavelength / (2 * max_resolution))
    )
    maximum_detector_distance = np.minimum(maximum_detector_distance, 75e-2)

    # Lower bound: keep the field of view large enough to contain several
    # texture periods. Very short distances are clipped at 0.03 m.
    min_fov = 10 * stripe_width
    min_resolution = min_fov / params["detector_shape"][0]
    minimum_detector_distance = detector_width / (
        2 * np.tan(wavelength / (2 * min_resolution))
    )
    minimum_detector_distance = np.maximum(minimum_detector_distance, 3e-2)

    if maximum_detector_distance <= minimum_detector_distance:
        return minimum_detector_distance
    return Uniform(minimum_detector_distance, maximum_detector_distance)


def random_aperture_config(params):
    """Generate one random OH/RH holography mask from sampled geometry.

    Parameters
    ----------
    params : Any
        Input value for ``params``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    # This function is also called once per simulated sample, after
    # detector_distance_range has already chosen a detector distance. It returns
    # all aperture lists together so their lengths and geometric constraints
    # stay consistent.
    wavelength = physics.photon_energy_wavelength(params["energy"])
    stripe_width = params["pattern_config"]["stripe_width"]
    detector_width = params["detector_shape"][0] * params["detector_pixel_size"]
    detector_distance = params["detector_distance"]

    # Convert the detector-limited real-space resolution into a total mask FOV.
    real_space_resolution = wavelength / (
        2 * np.arctan(detector_width / (2 * detector_distance))
    )
    fov_xy = params["detector_shape"][0] * real_space_resolution

    # Start with the object hole at the mask origin.
    aperture_types = ["OH"]
    aperture_centers = [(0.0, 0.0)]
    aperture_angles = [0.0]
    aperture_ellipticities = [1.0]
    aperture_seeds = [int(np.random.randint(0, 2**31 - 1))]
    aperture_top_radius_factors = [Uniform(1.0, 2.0).sample()]

    # Object hole radius: larger than the sampled texture period, but not so
    # large that a small magnetic texture length produces an enormous local
    # pattern-generation ROI.
    real_space_pixel_size = real_space_resolution / params["oversampling"]
    oh_radius_min = max(stripe_width, 250e-9)
    oh_radius_max = np.amin(
        [
            min(fov_xy / 8, 3e-6),
            3e-6,
            max_oh_radius_in_texture_widths * stripe_width,
            (
                max_magnetic_pattern_roi_pixels
                * real_space_pixel_size
                / (2.5 * aperture_top_radius_factors[0])
            ),
        ]
    )
    if oh_radius_max <= oh_radius_min:
        oh_radius = oh_radius_min
    else:
        oh_radius = Uniform(oh_radius_min, oh_radius_max).sample()

    aperture_radii = [oh_radius]
    aperture_sigmas = [Uniform(10e-9, 20e-9).sample()]
    oh_roughness, oh_roughness_modes = aperture_roughness_from_length(
        oh_radius,
        amplitude=aperture_roughness_amplitude,
        period=aperture_roughness_period,
    )
    aperture_roughnesses = [oh_roughness]
    aperture_roughness_modes = [oh_roughness_modes]

    # Add 1 to 5 reference holes. Each RH has its own radius, edge sigma, and
    # random position inside the FOV but outside 2 * OH_radius from the origin.
    n_reference_holes = np.random.randint(1, 6)
    aperture_top_radius_RH = 200e-9
    oh_top_radius = oh_radius * aperture_top_radius_factors[0]
    # The reference holes should be far from the OH autocorrelation and should
    # not overlap the OH or each other at the widest/top aperture opening.
    min_oh_distance = max(3 * oh_radius, aperture_top_radius_RH + oh_top_radius)
    # maximum distance from the center is set by the FOV, but we also want to
    center_half_width = np.abs(fov_xy / 2 - oh_radius)
    center_half_width = np.minimum(center_half_width,6*oh_radius)
    placed_reference_holes = []

    for _ in range(n_reference_holes):
        rh_radius = Uniform(5e-9, 75e-9).sample()
        rh_top_radius_factor = np.maximum(aperture_top_radius_RH, rh_radius*3)/rh_radius
        rh_top_radius = rh_radius * rh_top_radius_factor
        rh_roughness, rh_roughness_modes = aperture_roughness_from_length(
            rh_radius,
            amplitude=aperture_roughness_amplitude,
            period=aperture_roughness_period,
        )

        rh_sigma_max = max(1e-9, rh_radius / 4)
        if rh_sigma_max == 1e-9:
            rh_sigma = 1e-9
        else:
            rh_sigma = Uniform(1e-9, rh_sigma_max).sample()

        def _reference_hole_is_clear(candidate_y, candidate_x):
            """Handle the internal reference hole is clear operation.

            Parameters
            ----------
            candidate_y : Any
                Input value for ``candidate_y``.
            candidate_x : Any
                Input value for ``candidate_x``.

            Returns
            -------
            result : Any
                Return value produced by the function.
            """
            if np.hypot(candidate_y, candidate_x) <= min_oh_distance:
                return False
            for placed_y, placed_x, placed_top_radius in placed_reference_holes:
                min_distance = rh_top_radius + placed_top_radius
                if np.hypot(candidate_y - placed_y, candidate_x - placed_x) <= min_distance:
                    return False
                twin_distance = np.hypot(
                    np.abs(candidate_y) - np.abs(placed_y),
                    np.abs(candidate_x) - np.abs(placed_x),
                )
                if twin_distance <= min_distance:
                    return False
            return True

        for _attempt in range(1000):
            center_y = np.random.uniform(-center_half_width, center_half_width)
            center_x = np.random.uniform(-center_half_width, center_half_width)
            if _reference_hole_is_clear(center_y, center_x):
                break
        else:
            # Tight geometries can run out of room; try polar candidates, then
            # skip this RH rather than allowing aperture overlap.
            for _attempt in range(1000):
                center_radius = np.random.uniform(
                    min_oh_distance, max(min_oh_distance, 0.45 * fov_xy)
                )
                center_angle = np.random.uniform(0.0, 2 * np.pi)
                center_y = center_radius * np.sin(center_angle)
                center_x = center_radius * np.cos(center_angle)
                if _reference_hole_is_clear(center_y, center_x):
                    break
            else:
                continue

        aperture_types.append("RH")
        aperture_radii.append(rh_radius)
        aperture_centers.append((center_y, center_x))
        aperture_sigmas.append(rh_sigma)
        aperture_angles.append(Uniform(0.0, np.pi).sample())
        aperture_ellipticities.append(Uniform(0.65, 1.55).sample())
        aperture_roughnesses.append(rh_roughness)
        aperture_roughness_modes.append(rh_roughness_modes)
        aperture_seeds.append(int(np.random.randint(0, 2**31 - 1)))
        aperture_top_radius_factors.append(rh_top_radius_factor)
        placed_reference_holes.append((center_y, center_x, rh_top_radius))

    return {
        "aperture_types": aperture_types,
        "aperture_radii": aperture_radii,
        "aperture_centers": aperture_centers,
        "aperture_sigmas": aperture_sigmas,
        "aperture_angles": aperture_angles,
        "aperture_ellipticities": aperture_ellipticities,
        "aperture_roughnesses": aperture_roughnesses,
        "aperture_roughness_modes": aperture_roughness_modes,
        "aperture_seeds": aperture_seeds,
        "aperture_top_radius_factors": aperture_top_radius_factors,
    }


def random_coherence_length(params):
    """Sample similar y/x coherence lengths in metres.

    Parameters
    ----------
    params : Any
        Input value for ``params``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    base = Uniform(15e-6, 50e-6).sample()
    anisotropy = Uniform(-0.13, 0.13).sample()
    return (base * (1.0 + anisotropy), base * (1.0 - anisotropy))


def random_detector_center(params):
    """Sample an off-centre direct-beam position in detector pixels."""
    shape_y, shape_x = params["detector_shape"]
    return (
        Uniform(
            shape_y / 2 - detector_center_jitter_fraction * shape_y,
            shape_y / 2 + detector_center_jitter_fraction * shape_y,
        ).sample(),
        Uniform(
            shape_x / 2 - detector_center_jitter_fraction * shape_x,
            shape_x / 2 + detector_center_jitter_fraction * shape_x,
        ).sample(),
    )


def random_beamstop_config(params):
    """Sample beamstop geometry and offset it around the sampled detector centre.

    Beamstop radius is specified at the beamstop plane in metres. The centre is
    stored in detector pixels, so the random offset is drawn in units of the
    projected beamstop radius on the detector.
    """
    radius = Uniform(0.15e-3, 0.5e-3).sample()
    projected_radius_px = (
        radius
        * params["detector_distance"]
        / (params["detector_distance"] - beamstop_distance)
        / params["detector_pixel_size"]
    )
    max_offset_px = beamstop_center_jitter_radius_fraction * projected_radius_px
    center_offset_px = (
        Uniform(-max_offset_px, max_offset_px).sample(),
        Uniform(-max_offset_px, max_offset_px).sample(),
    )
    return {
        "radius": radius,
        "center_offset_px": center_offset_px,
        "angle": Uniform(0.0, np.pi).sample(),
        "sigma": Uniform(10e-6, 30e-6).sample(),
        "wire_width": Uniform(0.05e-3, 0.1e-3).sample(),
        "wire_bend": Uniform(0.0, 0.75e-3).sample(),
        "antialias": 2,
    }


###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

ranges = HologramPipelineRanges(
    # Sweep X-ray energy across the Co L-edge absorption region
    xray_energy=Uniform(775, 795),
    # Move the direct beam away from the exact detector middle by up to
    # +/-5% of the image size in y and x.
    detector_center=random_detector_center,

    xray_coherence_length=random_coherence_length,
    # Random illumination geometry
    illumination_focus_distance=Uniform(0.0, 2e-3),
    illumination_fwhm=Uniform(5e-6, 50e-6),
    illumination_center=(
        Uniform(-2e-6, 2e-6),
        Uniform(-2e-6, 2e-6),
    ),
    measurement_config=lambda params: {
        "number_frames": int(np.random.randint(1, 21)),
        "max_counts_per_image": Uniform(40_000, 70_000).sample(),
        "exposure_time": measurement_config["exposure_time"],
    },
    detector_params={
        # counts_per_photon belongs to detector_params because it controls the
        # conversion between detector counts and photon events.
        "counts_per_photon": Uniform(80, 220),
    },
    # Pattern-type mix: 40% labyrinth, 40% skyrmion lattice, 20% saturated.
    pattern_type=Choice(
        (
            "binary_labyrinth_pattern",
            "binary_labyrinth_pattern",
            "binary_labyrinth_pattern",
            "binary_labyrinth_pattern",
            "disordered_skyrmion_lattice_pattern",
            "disordered_skyrmion_lattice_pattern",
            "disordered_skyrmion_lattice_pattern",
            "saturated_pattern",
        )
    ),

    # Generate interdependent magnetic stripe parameters first.
    pattern_config=random_pattern_config,
    # Generate beamstop parameters from reasonable ranges. The beamstop centre
    # follows the sampled detector centre, with an additional random offset of
    # +/-0.5 projected beamstop radii in detector pixels.
    beamstop_config=random_beamstop_config,

    artifacts_config = {
        "sigma_photon": Uniform(0.7,0.9),
        "photon_n_classes": 1,
        "photon_n_variants": 30,
        "photon_kernel_size": 9,
        "photon_irregularity": 2.0,
        "regenerate_photon_kernels": True,
    },


    # generating detector distances from reasonable ranges based on the stripe width and xray energy
    # you can also choose a uniform distribution or a fixed value
    detector_distance=detector_distance_range,
    # generating holography masks from reasonable ranges based on the stripe width, xray energy and detector distance
    # you can also choose a uniform distribution or a fixed value
    aperture_config=random_aperture_config,
    # All other parameters use the fixed values from config above
)


###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

# %%
# ===================
# RUN PIPELINE
# ===================


pipeline = HologramPipeline(
    config=config,
    ranges=ranges,
    output_path=output_path,
    n_samples=nr_simulations,
    verbose=True,
)

pipeline.run()

# %%
# ===================
# INSPECT OUTPUT + FIGURE
# ===================
# Each group '00000/', '00001/', ... contains:
#
#   CR/ideal       — ideal (noise-free) hologram for circular-right polarisation
#   CR/detected    — detected (noisy) hologram
#   CR/detected_no_beamstop — optional detected hologram without beamstop shadow
#   CR/exit_wave   — complex scalar exit wavefield
#   CL/...         — same for circular-left
#   beamstop_mask  — 2D beamstop mask
#   metadata/      — all physical parameters used for this run
import matplotlib.pyplot as plt

PLOT_FIGSIZE_MAIN = (11, 8)
PLOT_FIGSIZE_APERTURE = (11, 6.5)
PLOT_FIGSIZE_RECONSTRUCTION = (9, 7)

with h5py.File(output_path, "r") as h5:
    grp = h5["00000"]
    if False:
        print(f"File contains {nr_simulations} simulations.")
        print(f"Top-level keys: {list(h5.keys())}\n")

        print("Datasets in '00000/':")
        grp.visit(lambda name: print(f"  {name}"))

        print("\nMetadata for '00000/':")

        def _print_meta(name, obj):
            """Handle the internal print meta operation.

            Parameters
            ----------
            name : Any
                Input value for ``name``.
            obj : Any
                Input value for ``obj``.

            Returns
            -------
            result : Any
                Return value produced by the function.
            """
            if hasattr(obj, "shape") and obj.shape == ():
                print(f"  {name} = {obj[()]}")

        grp["metadata"].visititems(_print_meta)

    # Load all arrays for the first simulation (exit wave kept complex)
    # Datasets have shape (N_frames, Ny, Nx); select one frame for display.
    frame = 0
    cr_exit     = grp["CR/exit_wave"][frame]
    cr_ideal    = grp["CR/ideal"][frame]
    cr_detected = grp["CR/detected"][frame]

    cl_exit     = grp["CL/exit_wave"][frame]
    cl_ideal    = grp["CL/ideal"][frame]
    cl_detected = grp["CL/detected"][frame]

    recipe_saved = grp["metadata/sample/recipe"][()].decode()
    #detector_shape_saved = tuple(grp["metadata/detector/shape"][()].astype(int))
    detector_shape_saved = tuple(ast.literal_eval(grp["metadata/detector/shape"][()].decode() if isinstance(grp["metadata/detector/shape"][()], bytes) else grp["metadata/detector/shape"][()]))
    oversampling_saved = int(h5["_pipeline_config/oversampling"][()])
    real_space_pixel_size_saved = float(
        grp["metadata/sample/real_space_pixel_size"][()]
    )
    parsed_recipe = structures.parse_recipe(recipe_saved)
    layer_names_saved = [layer.material for layer in parsed_recipe.layers]
    layer_thicknesses_saved = [layer.thickness for layer in parsed_recipe.layers]
    sample_shape_saved = (
        len(layer_names_saved),
        oversampling_saved * detector_shape_saved[0],
        oversampling_saved * detector_shape_saved[1],
    )
    membrane_index_saved = layer_names_saved.index("SiN")
    thickness_oh_saved = float(np.sum(layer_thicknesses_saved[:membrane_index_saved]))
    aperture_taper_depth_saved = float(
        np.sum(layer_thicknesses_saved[: max(0, membrane_index_saved - 2)])
    )
    aperture_meta = grp["metadata/sample/aperture/aperture_config"]
    aperture_types_saved = [
        t.decode() if isinstance(t, bytes) else str(t)
        for t in aperture_meta["apertures_type"][()]
    ]
    aperture_config_saved = dict(
        apertures_type=aperture_types_saved,
        apertures_radius=aperture_meta["apertures_radius"][()],
        apertures_center=aperture_meta["apertures_center"][()],
        apertures_sigma=aperture_meta["apertures_sigma"][()],
        apertures_angle=aperture_meta["apertures_angle"][()],
        apertures_ellipticity=aperture_meta["apertures_ellipticity"][()],
        apertures_roughness=aperture_meta["apertures_roughness"][()],
        apertures_roughness_modes=aperture_meta["apertures_roughness_modes"][()],
        apertures_seed=aperture_meta["apertures_seed"][()],
        apertures_top_radius_factor=aperture_meta[
            "apertures_top_radius_factor"
        ][()],
        aperture_taper_depth=aperture_taper_depth_saved,
        thickness_OH=thickness_oh_saved,
    )
    aperture_view = FrontApertureConfig(
        aperture_method="FTH_circular",
        aperture_shape=sample_shape_saved,
        real_space_pixel_size=real_space_pixel_size_saved,
        aperture_thicknesses=layer_thicknesses_saved,
        aperture_config=aperture_config_saved,
        use_roi=use_roi,
    )
    aperture_view.setup()
    aperture_mask = aperture_view.return_aperture()
    aperture_material_fraction = np.mean(aperture_mask, axis=0)
    aperture_cut_pixels = []
    for center in aperture_config_saved["apertures_center"]:
        cut_y = int(
            np.clip(
                round(sample_shape_saved[1] / 2 + float(center[0]) / real_space_pixel_size_saved),
                0,
                sample_shape_saved[1] - 1,
            )
        )
        cut_x = int(
            np.clip(
                round(sample_shape_saved[2] / 2 + float(center[1]) / real_space_pixel_size_saved),
                0,
                sample_shape_saved[2] - 1,
            )
        )
        aperture_cut_pixels.append((cut_y, cut_x))
    aperture_yz_cuts = np.stack(
        [aperture_mask[:, :, cut_x] for _, cut_x in aperture_cut_pixels]
    )
    aperture_xz_cuts = np.stack(
        [aperture_mask[:, cut_y, :] for cut_y, _ in aperture_cut_pixels]
    )

if False:
    print(f"Hologram shape: {cr_ideal.shape}")

# ------------------------------------------------------------------
# Helper: colour limits matching HologramConfig.visualize_averages()
# ------------------------------------------------------------------
def _clim(arr):
    """Handle the internal clim operation.

    Parameters
    ----------
    arr : Any
        Input value for ``arr``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    return np.nanpercentile(arr, [0.1, 99.9])

def _sym_clim(diff):
    """Handle the internal sym clim operation.

    Parameters
    ----------
    diff : Any
        Input value for ``diff``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    d = np.nanpercentile(np.abs(diff), 99.9)
    return -d, d

# ------------------------------------------------------------------
# Figure 1 — CR, CL, CR−CL difference
#
# Layout (4 rows × 3 cols), mirrors HologramConfig.visualize_averages():
#
#   Row 0  Exit wave amplitude   inferno    [0.1–99.9 pct]
#   Row 1  Exit wave phase       hsv        [−π, π]
#   Row 2  Ideal hologram        viridis    [0.1–99.9 pct]
#   Row 3  Detected hologram     viridis    [0.1–99.9 pct]
#   Cols   CR  |  CL  |  CR−CL (RdBu_r, symmetric ±99.9 pct)
# ------------------------------------------------------------------
cr_amp, cl_amp = np.abs(cr_exit), np.abs(cl_exit)
cr_pha, cl_pha = np.angle(cr_exit), np.angle(cl_exit)

rows_spec = [
    # (cr_data,    cl_data,    cmap,      label,               cbar_label)
    (cr_amp,     cl_amp,     "inferno", "Exit wave |E|",      "amplitude"),
    (cr_pha,     cl_pha,     "hsv",     "Exit wave phase",    "rad"),
    (cr_ideal,   cl_ideal,   "viridis", "Ideal hologram",     "counts"),
    (cr_detected, cl_detected, "viridis", "Detected hologram", "counts"),
]

fig1, axes1 = plt.subplots(4, 3, figsize=PLOT_FIGSIZE_MAIN, constrained_layout=True)
fig1.suptitle("Simulation 00000 — CR, CL, CR−CL difference", fontsize=13)

for row_idx, (cr_d, cl_d, cmap, label, cbar_label) in enumerate(rows_spec):
    diff = cr_d - cl_d

    if cmap == "hsv":
        lo_cr, hi_cr = -np.pi, np.pi
        lo_cl, hi_cl = -np.pi, np.pi
    else:
        lo_cr, hi_cr = _clim(cr_d)
        lo_cl, hi_cl = _clim(cl_d)

    diff_lo, diff_hi = _sym_clim(diff)

    panels = [
        (cr_d,  f"CR — {label}",    lo_cr,    hi_cr,    cmap),
        (cl_d,  f"CL — {label}",    lo_cl,    hi_cl,    cmap),
        (diff,  f"CR−CL — {label}", diff_lo,  diff_hi,  "RdBu_r"),
    ]

    for col_idx, (img, title, lo, hi, cm) in enumerate(panels):
        ax = axes1[row_idx, col_idx]
        m = ax.imshow(
            img,
            cmap=cm,
            vmin=lo,
            vmax=hi,
            origin="upper",
            interpolation="none",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        fig1.colorbar(m, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

plt.show()

# ------------------------------------------------------------------
# Figure 2 — Aperture depth profile sanity check
#
# Rebuilds the aperture mask from metadata for plotting only. Nothing extra is
# saved in the HDF5 file.
# ------------------------------------------------------------------
nr_aperture_plots = min(len(aperture_types_saved), 4)
fig2, axes2 = plt.subplots(
    2,
    1 + nr_aperture_plots,
    figsize=PLOT_FIGSIZE_APERTURE,
    constrained_layout=True,
)
fig2.suptitle("Aperture depth profiles — OH/RH cuts", fontsize=12)

axes2[0, 0].axis("off")
m = axes2[1, 0].imshow(
    aperture_material_fraction,
    cmap="gray",
    vmin=0,
    vmax=1,
    origin="upper",
    aspect="equal",
    interpolation="none",
)
axes2[1, 0].set_title("Depth-averaged material", fontsize=10)
axes2[1, 0].set_xlabel("x (px)")
axes2[1, 0].set_ylabel("y (px)")
fig2.colorbar(m, ax=axes2[1, 0], fraction=0.046, pad=0.04, label="material fraction")

for i in range(nr_aperture_plots):
    label = f"{aperture_types_saved[i]} {i}"
    for row, data, xlabel in (
        (0, aperture_yz_cuts[i], "y (px)"),
        (1, aperture_xz_cuts[i], "x (px)"),
    ):
        ax = axes2[row, i + 1]
        title_axis = "Y-Z" if row == 0 else "X-Z"
        m = ax.imshow(
            data,
            cmap="gray",
            vmin=0,
            vmax=1,
            origin="upper",
            aspect="auto",
            interpolation="none",
        )
        ax.set_title(f"{title_axis} cut {label}", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("layer index")
        fig2.colorbar(m, ax=ax, fraction=0.046, pad=0.04, label="material fraction")

if nr_aperture_plots == 0:
    for ax in axes2.flat:
        ax.axis("off")
else:
    for j in range(nr_aperture_plots + 1, axes2.shape[1]):
        axes2[0, j].axis("off")
        axes2[1, j].axis("off")

plt.show()

# ------------------------------------------------------------------
# Figure 3 — FTH reconstruction of the ideal CR−CL difference
#
# Mirrors HologramConfig.visualize_reconstruction(source="ideal",
#                                                  helicity="diff")
# Reconstruction: fftshift(fft2(fftshift(hologram)))
# ------------------------------------------------------------------
diff_ideal = cr_ideal - cl_ideal
rec = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(diff_ideal)))

rec_panels = [
    (np.abs(rec),   "Amplitude",  "inferno", None,   None,   "Amplitude"),
    (np.angle(rec), "Phase",      "hsv",     -np.pi, np.pi,  "Phase (rad)"),
    (np.real(rec),  "Real",       "gray",    None,   None,   "Real part"),
    (np.imag(rec),  "Imaginary",  "gray",    None,   None,   "Imag part"),
]

fig3, axes3 = plt.subplots(
    2,
    2,
    figsize=PLOT_FIGSIZE_RECONSTRUCTION,
    constrained_layout=True,
)
fig3.suptitle(
    "FTH reconstruction — ideal CR−CL difference (simulation 00000)", fontsize=12
)

for ax, (data, title, cmap, vmin, vmax, cbar_label) in zip(axes3.flat, rec_panels):
    if vmin is None:
        vmin, vmax = np.nanpercentile(data, [0.1, 99.9])
    m = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="none",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig3.colorbar(m, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

plt.show()

# %%
