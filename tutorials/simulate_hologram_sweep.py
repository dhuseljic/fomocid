# %%
# Import general libraries
import os
import numpy as np
import h5py
from scattering_calculator.utils import physics

# Imports from our own codebase
from fomocid import DATA_ROOT
from scattering_calculator.simulation_pipelines import (
    Uniform,
    Choice,
)
from scattering_calculator.simulation_pipelines.pipelines import (
    HologramPipeline,
    HologramPipelineConfig,
    HologramPipelineRanges,
)


#################################################################
#### HOW MANY SIMULATIONS TO RUN? ####
#################################################################
nr_simulations = 1  # increase to e.g. 1000 for a full training dataset



# %%
# ===================
# OUTPUT PATH
# ===================
output_folder = DATA_ROOT / "Data" / "hologram_sweep"
output_path = output_folder / "simulation_sweep.h5"
pipeline_random_seed = None  # set to None for non-reproducible random sweeps
use_roi = True
dielectric_tensor_use_roi = True

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
detector_center = (650, 650)  # px
detector_params = {
    "readout_noise_average": 50,
    "noise_rms": 3,
    "detector_threshold": 64.3e3,
    "counts_per_photon": 100,
    "quantum_efficiency": 1.0,
}
measurement_config = {
    "number_frames": 1,
    "max_counts_per_image": 64.3e3,
    "exposure_time": 1e-2,
}
artifacts_config = {
    "counts_per_photon": 100,
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
    "seed": None,
}

# --- Material stack ---
recipe = "[Au(70)/Cr(30)]x5/SiN(200)/Co(90)/Pt(120)/Al(60)"

# --- Magnetic domain pattern ---
pattern_type = "binary_labyrinth_pattern"  # "wavy_stripe_pattern", "binary_labyrinth_pattern", "disordered_skyrmion_lattice_pattern", or "saturated_pattern"
stripe_width = 300e-9  # m
sigma = 30e-9  # m
angle_stripes = np.pi / 4
waviness_amplitude = 0e-9  # ms
waviness_scale = 0e-9  # m
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
else:
    raise ValueError(f"Unknown pattern_type: {pattern_type}")

# --- FTH holography mask ---
aperture_types = ["OH", "RH", "RH"]
aperture_radii = [60e-9, 6e-9, 4e-9]  # m
aperture_centers = [(0, 0), (0.2e-6, -0.15e-6), (0.15e-6, 0.15e-6)]  # m (y, x)
aperture_sigmas = [1e-9, 2e-9, 2e-9]  # m
aperture_angles = [0.0, 0.0, 0.0]  # rad
aperture_ellipticities = [1.0, 1.0, 1.0]  # y/x axis ratio
aperture_roughnesses = [0.0, 0.04, 0.04]
aperture_roughness_modes = [(0, 0), (3, 9), (3, 9)]
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
    # Beamstop
    beamstop_method=beamstop_method,
    beamstop_distance=beamstop_distance,
    beamstop_config=beamstop_config,
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
    # Simulation grid: sample_shape = oversampling * detector_shape
    oversampling=2,
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
    """Generate interdependent magnetic-pattern parameters in physical units."""
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
    """Choose a detector-distance range from sampled energy and stripe width."""
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
    """Generate one random OH/RH holography mask from sampled geometry."""
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
    aperture_roughnesses = [0.0]
    aperture_roughness_modes = [(0, 0)]
    aperture_seeds = [-1]
    aperture_top_radius_factors = [2.0]

    # Object hole radius: larger than the sampled texture period and 100 nm,
    # but smaller than one quarter of the mask FOV and 5 um.
    oh_radius_min = max(stripe_width, 250e-9)
    oh_radius_max = min(fov_xy / 8, 3e-6)
    if oh_radius_max <= oh_radius_min:
        oh_radius = oh_radius_min
    else:
        oh_radius = Uniform(oh_radius_min, oh_radius_max).sample()

    aperture_radii = [oh_radius]
    aperture_sigmas = [Uniform(10e-9, 20e-9).sample()]

    # Add 1 to 5 reference holes. Each RH has its own radius, edge sigma, and
    # random position inside the FOV but outside 2 * OH_radius from the origin.
    n_reference_holes = np.random.randint(1, 6)
    aperture_top_radius_RH = 200e-9
    #the ref oles should be at least 3 times the OH radius
    #away from the center to avoid autocorrelation overlap. also avoid cone overlap by ensuring the RH top radius doesn't overlap with the OH top radius at the center, which is the worst case for cone overlap.
    min_center_distance = 3 * oh_radius
    min_center_distance = np.maximum( min_center_distance, aperture_top_radius_RH+oh_radius* aperture_top_radius_factors[0])
    # maximum distance from the center is set by the FOV, but we also want to
    center_half_width = np.abs(fov_xy / 2 - oh_radius)
    center_half_width = np.minimum(center_half_width,6*oh_radius)

    for _ in range(n_reference_holes):
        aperture_types.append("RH")

        rh_radius = Uniform(5e-9, 90e-9).sample()
        aperture_radii.append(rh_radius)
        aperture_angles.append(Uniform(0.0, np.pi).sample())
        aperture_ellipticities.append(Uniform(0.65, 1.55).sample())
        aperture_roughnesses.append(Uniform(0.01, 0.08).sample())
        aperture_roughness_modes.append((3, 9))
        aperture_seeds.append(int(np.random.randint(0, 2**31 - 1)))
        aperture_top_radius_factors.append(np.maximum(aperture_top_radius_RH, rh_radius*3)/rh_radius)

        rh_sigma_max = max(1e-9, rh_radius / 4)
        if rh_sigma_max == 1e-9:
            aperture_sigmas.append(1e-9)
        else:
            aperture_sigmas.append(Uniform(1e-9, rh_sigma_max).sample())

        for _attempt in range(1000):
            center_y = np.random.uniform(-center_half_width, center_half_width)
            center_x = np.random.uniform(-center_half_width, center_half_width)
            if np.hypot(center_y, center_x) > min_center_distance:
                break
        else:
            # Extremely unlikely fallback for very tight geometries.
            center_radius = min(0.45 * fov_xy, 1.05 * min_center_distance)
            center_angle = np.random.uniform(0.0, 2 * np.pi)
            center_y = center_radius * np.sin(center_angle)
            center_x = center_radius * np.cos(center_angle)

        aperture_centers.append((center_y, center_x))

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
    """Sample similar y/x coherence lengths in metres."""
    base = Uniform(5e-6, 50e-6).sample()
    anisotropy = Uniform(-0.13, 0.13).sample()
    return (base * (1.0 + anisotropy), base * (1.0 - anisotropy))


###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

ranges = HologramPipelineRanges(
    # Sweep X-ray energy across the Co L-edge absorption region
    xray_energy=Uniform(775, 795),
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
    # Pattern-type mix: 40% labyrinth, 40% skyrmion lattice, 20% saturated.
    pattern_type=Choice(
        (
            "binary_labyrinth_pattern",
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
    # Generate beamstop parameters from reasonable ranges
    beamstop_config={
        "radius": Uniform(0.1e-3, 0.5e-3),
        "angle": Uniform(0.0, np.pi),
        "sigma": Uniform(10e-6, 30e-6),
        "wire_width": Uniform(0.05e-3, 0.1e-3),
        "wire_bend": Uniform(0.0, 0.75e-3),
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
#   CR/exit_wave   — complex scalar exit wavefield
#   CL/...         — same for circular-left
#   beamstop_mask  — 2D beamstop mask
#   metadata/      — all physical parameters used for this run
import matplotlib.pyplot as plt

with h5py.File(output_path, "r") as h5:
    grp = h5["00000"]
    if False:
        print(f"File contains {nr_simulations} simulations.")
        print(f"Top-level keys: {list(h5.keys())}\n")

        print("Datasets in '00000/':")
        grp.visit(lambda name: print(f"  {name}"))

        print("\nMetadata for '00000/':")

        def _print_meta(name, obj):
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
    #aperture_material_fraction = grp["aperture_material_fraction"][()]
    #aperture_yz_cut = grp["aperture_yz_cut"][()]
    #aperture_xz_cut = grp["aperture_xz_cut"][()]
    #aperture_yz_cuts = grp["aperture_yz_cuts"][()]
    #aperture_xz_cuts = grp["aperture_xz_cuts"][()]
    aperture_types_saved = [
        t.decode() if isinstance(t, bytes) else str(t)
        for t in grp["metadata/aperture/aperture_config/apertures_type"][()]
    ]

if False:
    print(f"Hologram shape: {cr_ideal.shape}")

# ------------------------------------------------------------------
# Helper: colour limits matching HologramConfig.visualize_averages()
# ------------------------------------------------------------------
def _clim(arr):
    return np.nanpercentile(arr, [0.1, 99.9])

def _sym_clim(diff):
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

fig1, axes1 = plt.subplots(4, 3, figsize=(13, 17), constrained_layout=True)
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
        m = ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, origin="upper")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        fig1.colorbar(m, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

plt.show()

if False:
# ------------------------------------------------------------------
    # Figure 2 — Aperture geometry sanity check
    #
    # aperture_material_fraction is the depth-averaged material mask:
    #   1 = material remains through the stack
    #   0 = fully drilled away through the stack
    #
    # aperture_yz_cuts and aperture_xz_cuts are material masks through every
    # OH/RH centre. They should show the conical taper from wide top opening to
    # the nominal bottom aperture.
    # ------------------------------------------------------------------
    nr_aperture_plots = min(len(aperture_types_saved), 4)
    fig2, axes2 = plt.subplots(
        2,
        1 + nr_aperture_plots,
        figsize=(4.0 * (1 + nr_aperture_plots), 7.5),
        constrained_layout=True,
    )
    fig2.suptitle("Aperture geometry — average material and OH/RH cuts", fontsize=12)

    axes2[0, 0].axis("off")
    m = axes2[1, 0].imshow(
        aperture_material_fraction,
        cmap="gray",
        vmin=0,
        vmax=1,
        origin="upper",
        aspect="equal",
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
            )
            ax.set_title(f"{title_axis} cut {label}", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("layer index")
            fig2.colorbar(
                m, ax=ax, fraction=0.046, pad=0.04, label="material fraction"
            )

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

fig3, axes3 = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
fig3.suptitle(
    "FTH reconstruction — ideal CR−CL difference (simulation 00000)", fontsize=12
)

for ax, (data, title, cmap, vmin, vmax, cbar_label) in zip(axes3.flat, rec_panels):
    if vmin is None:
        vmin, vmax = np.nanpercentile(data, [0.1, 99.9])
    m = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig3.colorbar(m, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

plt.show()

# %%
