"""Shared physical magnetic-pattern names and legacy generator translation."""
import numpy as np

# Alias -> (shared name, multiplier). All lengths here are metres.
ALIASES = {
    'wavy_stripe_pattern': {'stripe_width':('period',2.), 'angle_stripes':('angle',1.)},
    'binary_labyrinth_pattern': {'stripe_width':('period',2.)},
    'skyrmion_pattern': {'skyr_radius':('radius',1.), 'screening_radius':('min_separation',2.)},
    'disordered_skyrmion_lattice_pattern': {'stripe_width':('radius',.5), 'diameter_spread':('radius_spread',.5)},
    'neel_lattice': {'lattice_spacing':('period',1.)},
    'magnon_wave': {'wave_angle':('angle',1.), 'oop_amplitude':('amplitude',1.)},
}


def canonical_parameters(method, parameters, legacy=None):
    """Resolve shared physical names without mutating the supplied dictionaries.

    Parameters
    ----------
    method : str
        Selected magnetic-pattern generator.
    parameters : dict
        Pattern controls in metres and radians, or dimensionless amplitudes.
    legacy : dict or None
        Older physical-length configuration, accepted for compatibility.

    Returns
    -------
    dict
        Shared names with consistent values; conflicting aliases raise an error.
    """
    result = dict(parameters)
    for key, value in (legacy or {}).items():
        if key in result and not np.array_equal(result[key], value):
            raise ValueError(f'Conflicting magnetic-pattern value for {key!r}: use only pattern_config. pattern_config_length is a legacy alias.')
        result[key] = value
    for alias, (name, factor) in ALIASES.get(method, {}).items():
        if alias not in result:
            continue
        value = result.pop(alias) * factor
        if name in result and not np.allclose(result[name], value, rtol=1e-12, atol=0):
            raise ValueError(f'Conflicting magnetic-pattern values: {name!r} and legacy {alias!r}')
        result[name] = value
    for name in ('period', 'radius', 'min_separation'):
        if name in result and (not np.isfinite(result[name]) or result[name] <= 0):
            raise ValueError(f'{name} must be finite and positive')
    for name in ('angle', 'amplitude', 'radius_spread'):
        if name in result and not np.isfinite(result[name]):
            raise ValueError(f'{name} must be finite')
    return result


def generator_parameters(method, parameters):
    """Translate shared physical names at the legacy generator boundary.

    Parameters
    ----------
    method : str
        Selected low-level generator.
    parameters : dict
        Canonical physical pattern controls.

    Returns
    -------
    dict
        Generator keyword names, still in physical units before pixel conversion.
    """
    result = dict(parameters)
    for alias, (name, factor) in ALIASES.get(method, {}).items():
        if name in result:
            result[alias] = result.pop(name) / factor
    return result


def aperture_placement_defaults(parameters, aperture, shape, pixel_size):
    """Place skyrmion centers beyond the OH; let the aperture clip the texture.

    Parameters use physical lengths, except placement_center in row/column pixels.
    Accept both public aperture dictionaries and legacy pipeline dictionaries.
    Explicit placement settings are preserved. Without OH geometry, leave the
    generator's full-field behavior intact.
    """
    result = dict(parameters)
    types = aperture.get("apertures_type", aperture.get("aperture_types", []))
    if "OH" not in types:
        return result
    index = types.index("OH")
    radii = aperture.get("apertures_radius", aperture.get("aperture_radii"))
    centers = aperture.get("apertures_center", aperture.get("aperture_centers"))
    factors = aperture.get("apertures_top_radius_factor",
                           aperture.get("aperture_top_radius_factors", [2.] * len(radii)))
    center = centers[index]
    result.setdefault("placement_center", (shape[0] / 2 + center[0] / pixel_size,
                                          shape[1] / 2 + center[1] / pixel_size))
    diameter = 2 * result["radius"] if "radius" in result else result.get("stripe_width", 0.)
    result.setdefault("placement_radius", radii[index] * max(1., factors[index]) + diameter)
    return result


def sample_constrained_domain_parameters(rng, *, constrain_magnetization=False, target_magnetization=None, saturation_probability=.1):
    """Sample diverse physical domain controls without requesting state labels.

    Lengths are metres. Preparation and relaxation controls are independent of
    the measured state. By default target_magnetization=None leaves sign area
    unconstrained. constrain_magnetization=True samples Gaussian numeric targets (mean 0,
    FWHM 0.5, conditioned on [-0.9,0.9]); an
    explicit numeric target_magnetization fixes the target across samples.
    Otherwise saturation_probability (default 0.1) selects exact +/-1 states
    with equal polarity probabilities, including in free-magnetization mode.
    Half the draws use a smooth isotropic preparation: random initialization,
    zero disorder/alignment/order, stiffness 2, 400 steps and time step 0.2.
    Physical repeat is log-uniform from 64 to 1000 nm, without later resizing.
    For the remaining diverse preparations, alignment is zero for half the draws; other draws use an exponential
    with scale 0.5 conditioned on [0,24]. Spatial disorder is uniform [0,0.8],
    and initial order uniform [0,1], covering ordered and disordered preparations.
    Generate one relaxed realization, then measure it; spacing calibration is
    disabled. Pass a NumPy Generator to make the sweep reproducible.
    """
    if not np.isfinite(saturation_probability) or not 0 <= saturation_probability <= 1:
        raise ValueError("saturation_probability must be finite in [0,1]")
    length = float(np.exp(rng.uniform(np.log(64e-9), np.log(1000e-9))))
    preparation = str(rng.choice(('random', 'stripes', 'hexagonal', 'worms')))
    magnetization = target_magnetization
    if magnetization is None and constrain_magnetization:
        sigma_m = .5 / (2 * np.sqrt(2 * np.log(2)))
        magnetization = float(rng.normal(0., sigma_m))
        while not -.9 <= magnetization <= .9:
            magnetization = float(rng.normal(0., sigma_m))
    if target_magnetization is None and rng.random() < saturation_probability:
        magnetization = float(rng.choice((-1., 1.)))
    if magnetization is not None:
        magnetization = float(magnetization)
        if not np.isfinite(magnetization) or not -1 <= magnetization <= 1:
            raise ValueError("target_magnetization must be None or finite in [-1,1]")
    alignment = 0.
    if rng.random() >= .5:
        alignment = float(-.5 * np.log1p(-rng.random() * (-np.expm1(-48.))))
    parameters = dict(
        characteristic_length=length, target_magnetization=magnetization,
        alignment=alignment,
        direction=float(rng.uniform(0., np.pi)),
        disorder=float(rng.uniform(0., .8)),
        disorder_length=float(rng.uniform(0., .15) * length),
        initial_state=preparation, initial_order=float(rng.uniform(0., 1.)),
        initial_domain_count=int(rng.integers(1, 4)),
        worm_length=float(rng.uniform(1., 4.) * length),
        worm_waviness=float(rng.uniform(0., .3) * length),
        wall_stiffness=float(rng.uniform(1., 4.)),
        n_steps=int(rng.choice((100, 300, 600))),
        time_step=float(rng.choice((.05, .1, .2))),
        convergence_tolerance=float(10 ** rng.uniform(-5., -3.)),
        length_tolerance=.15,  # diagnostic only; no retries to hit a measured spacing
        max_length_iterations=1,
        sigma=float(rng.uniform(.025, .06) * length),
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    # Notebook 22-style unpinned, isotropic relaxation supports smooth labyrinths.
    # Keep magnetization/physical spacing independent of this preparation choice.
    if rng.random() < .5:
        parameters.update(alignment=0., disorder=0., initial_state='random',
                          initial_order=0., wall_stiffness=2., n_steps=400,
                          time_step=.2)
    return parameters


def constrained_sampling_pitch(parameters):
    """Return the maximum native pixel pitch for physical constrained controls.

    Parameters
    ----------
    parameters : dict
        Physical characteristic_length, optional sigma and target_magnetization.

    Returns
    -------
    float
        Maximum pitch in metres: 16 pixels per repeat, 2 per positive wall sigma,
        and 4 across the conservative minority stripe-width estimate. This last
        estimate is a sampling guard, not a predicted relaxed bubble diameter.
    """
    length = float(parameters['characteristic_length'])
    sigma = float(parameters.get('sigma') or 0.)
    magnetization = parameters.get('target_magnetization')
    if not np.isfinite(length) or length <= 0 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError('characteristic_length must be positive and sigma nonnegative and finite')
    if magnetization is not None and (not np.isfinite(magnetization) or not -1 <= magnetization <= 1):
        raise ValueError('target_magnetization must be finite and in [-1,1]')
    pitch = length / 16
    if sigma:
        pitch = min(pitch, sigma / 2)
    fraction = 0. if magnetization is None else (1 - abs(magnetization)) / 2
    if fraction > 0:
        pitch = min(pitch, length * fraction / 4)
    return pitch


def sample_sweep_domain_parameters(rng, experiment_index=None, *, free_probability=.48,
                                   gaussian_probability=.48, saturation_probability=.04):
    """Sample a mixture of free, Gaussian-target and saturated magnetic parents.

    Parameters
    ----------
    rng : numpy.random.Generator
        Reproducible parent parameter generator.
    experiment_index : int or None
        Parent index accepted for pipeline callers; no periodic schedule is used.
    free_probability, gaussian_probability, saturation_probability : float
        Nonnegative probabilities summing to one. Defaults are 0.48, 0.48, 0.04.
        Gaussian targets have mean zero, FWHM 0.5, and bounds [-0.9,0.9].

    Returns
    -------
    dict
        Physical controls with None, a Gaussian target, or exactly +/-1.
        Saturated polarity is chosen with equal probability.
    """
    probabilities = np.asarray([free_probability, gaussian_probability, saturation_probability], dtype=float)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.):
        raise ValueError('Magnetization probabilities must be nonnegative and sum to one')
    mode = rng.choice(('free', 'gaussian', 'saturated'), p=probabilities)
    target = float(rng.choice((-1., 1.))) if mode == 'saturated' else None
    return sample_constrained_domain_parameters(rng,
        constrain_magnetization=mode == 'gaussian', target_magnetization=target,
        saturation_probability=0.)


def compatible_detector_distance(distance, minimum, *, wavelength, detector_shape,
                                 pixel_size, center, target_pitch, max_oversampling,
                                 ignore_curvature=False, sample_feasible=False):
    """Find the feasible interval and optionally sample it directly.

    With sample_feasible=True, distance is an upper bound and every draw is
    uniform over the compatible interval. Otherwise retain feasible distances.
    Return None if even the minimum distance cannot resolve the sampled texture.
    Evaluate extrema of the same discrete qx coordinates used by DetectorConfig,
    without allocating detector-sized arrays during the bounded search.
    """
    if not np.isfinite([distance, minimum]).all() or not 0 < minimum <= distance:
        raise ValueError('Compatible-distance minimum must be positive and no greater than the sampled distance')
    if max_oversampling < 1:
        return None
    x = (np.array([0, detector_shape[1]-1])-center[1])*pixel_size
    nearest_y = np.clip(np.rint(center[0]), 0, detector_shape[0]-1)
    y = (np.array([0, detector_shape[0]-1, nearest_y])-center[0])*pixel_size

    def pitch(z):
        q = x[None, :]/z if ignore_curvature else x[None, :]/np.sqrt(z*z+x[None, :]**2+y[:, None]**2)
        return wavelength/np.ptp(q)/max_oversampling

    if pitch(distance) <= target_pitch:
        return float(np.random.uniform(minimum, distance)) if sample_feasible else float(distance)
    if pitch(minimum) > target_pitch:
        return None
    low, high = minimum, distance
    for _ in range(48):
        middle = (low+high)/2
        if pitch(middle) <= target_pitch:
            low = middle
        else:
            high = middle
    # Keep a tiny margin below the integer oversampling boundary.
    upper = max(minimum, low*(1-1e-12))
    return float(np.random.uniform(minimum, upper))
