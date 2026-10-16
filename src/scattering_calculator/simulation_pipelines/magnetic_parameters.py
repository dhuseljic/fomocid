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
