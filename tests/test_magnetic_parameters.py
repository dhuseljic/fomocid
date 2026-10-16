"""Shared magnetic controls preserve physical scales and reject ambiguity."""
from unittest.mock import patch
import numpy as np
import pytest
from scattering_calculator.simulation_pipelines import MagneticPatternConfig


@pytest.mark.parametrize('method,old,new',[
    ('wavy_stripe_pattern',{'stripe_width':20e-9,'angle_stripes':.3,'sigma':1e-9}, {'period':40e-9,'angle':.3,'sigma':1e-9}),
    ('binary_labyrinth_pattern',{'stripe_width':20e-9},{'period':40e-9}),
    ('skyrmion_pattern',{'skyr_radius':5e-9,'screening_radius':12e-9},{'radius':5e-9,'min_separation':24e-9}),
    ('disordered_skyrmion_lattice_pattern',{'stripe_width':20e-9,'diameter_spread':4e-9},{'radius':10e-9,'radius_spread':2e-9}),
    ('neel_lattice',{'lattice_spacing':40e-9,'radius':5e-9},{'period':40e-9,'radius':5e-9}),
    ('magnon_wave',{'period':40e-9,'oop_amplitude':.1,'wave_angle':.3},{'period':40e-9,'amplitude':.1,'angle':.3}),
])
def test_shared_names_resolve_to_the_same_physical_controls(method,old,new):
    """Equivalent aliases describe identical physical controls.

    Parameters
    ----------
    method : str
        Pattern type.
    old, new : dict
        Equivalent legacy and common parameter declarations.

    Returns
    -------
    None
        Raises if a physical scale changes during alias translation.
    """
    a=MagneticPatternConfig(pattern_type_method=method,pattern_config=old)
    b=MagneticPatternConfig(pattern_type_method=method,pattern_config=new)
    assert a.resolved_pattern_config()==b.resolved_pattern_config()==new


def test_period_is_a_full_stripe_repeat_and_is_converted_once_to_pixels():
    """A full 40 nm period gives a 10 pixel single stripe at 2 nm pitch.

    Parameters
    ----------
    None
        Uses a mocked low-level generator to inspect the physical boundary.

    Returns
    -------
    None
        Raises if period is treated as a single stripe or converted twice.
    """
    c=MagneticPatternConfig(pattern_type_method='wavy_stripe_pattern',shape=(32,32),real_space_pixel_size=2e-9,
        pattern_config={'period':40e-9,'angle':.2,'sigma':4e-9})
    with patch('scattering_calculator.sample_generator.pattern_generator.create_wavy_stripe_pattern',return_value=(np.zeros((32,32)),None)) as make:
        c.create_pattern()
    assert make.call_args.kwargs['stripe_width']==10.
    assert make.call_args.kwargs['angle_stripes']==.2
    assert make.call_args.kwargs['sigma']==2.
    assert c.pattern_config=={'period':40e-9,'angle':.2,'sigma':4e-9}


@pytest.mark.parametrize('method,parameters',[
    ('wavy_stripe_pattern',{'period':40e-9,'stripe_width':30e-9}),
    ('magnon_wave',{'amplitude':.1,'oop_amplitude':.2}),
    ('neel_lattice',{'period':40e-9,'lattice_spacing':30e-9}),
])
def test_conflicting_names_fail_instead_of_choosing_one(method,parameters):
    """Conflicting common and legacy values cannot silently alter the specimen.

    Parameters
    ----------
    method : str
        Pattern type.
    parameters : dict
        Deliberately conflicting aliases.

    Returns
    -------
    None
        Raises unless ambiguous settings are rejected.
    """
    with pytest.raises(ValueError,match='Conflicting'):
        MagneticPatternConfig(pattern_type_method=method,pattern_config=parameters).resolved_pattern_config()


def test_actual_rotated_stripes_and_magnons_are_unchanged_by_parameter_migration():
    """Independent legacy/new declarations produce identical vector and stripe maps.

    Parameters
    ----------
    None
        Uses seeded stripe and deterministic magnon generators.

    Returns
    -------
    None
        Raises if a renamed control changes the generated specimen.
    """
    for method,old,new in [
        ('wavy_stripe_pattern',{'stripe_width':12e-9,'angle_stripes':.3,'sigma':2e-9,'seed':4},
         {'period':24e-9,'angle':.3,'sigma':2e-9,'seed':4}),
        ('magnon_wave',{'period':40e-9,'oop_amplitude':.1,'wave_angle':.3},
         {'period':40e-9,'amplitude':.1,'angle':.3}),
        ('neel_lattice',{'lattice_spacing':24e-9,'radius':5e-9},
         {'period':24e-9,'radius':5e-9}),
        ('disordered_skyrmion_lattice_pattern',{'stripe_width':12e-9,'diameter_spread':2e-9,'sigma':1e-9,'seed':4},
         {'radius':6e-9,'radius_spread':1e-9,'sigma':1e-9,'seed':4}),
        ('skyrmion_pattern',{'skyr_radius':5e-9,'screening_radius':12e-9,'number_skyr':2,'number_iter':100,'seed':4},
         {'radius':5e-9,'min_separation':24e-9,'number_skyr':2,'number_iter':100,'seed':4}),
    ]:
        a=MagneticPatternConfig(pattern_type_method=method,shape=(48,48),real_space_pixel_size=2e-9,pattern_config=old)
        b=MagneticPatternConfig(pattern_type_method=method,shape=(48,48),real_space_pixel_size=2e-9,pattern_config=new)
        np.testing.assert_array_equal(a.create_pattern()[0],b.create_pattern()[0])
