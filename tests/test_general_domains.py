"""General scalar-domain controls and shared experiment configuration."""
import numpy as np
import pytest
from scattering_calculator.sample_generator.pattern_generator import create_domain_pattern, create_binary_labyrinth_pattern
from scattering_calculator.sample_generator.domain_catalogue import measure_domain_pattern
from scattering_calculator.simulation_pipelines.simulation_configuration import MagneticPatternConfig


def test_general_defaults_recover_existing_labyrinth():
    common = dict(n_steps=30, seed=7, use_gpu=False, noise_amp=0., sigma=.7, domain_conversion="soft")
    actual, _ = create_domain_pattern((64,64), period=16, rescale_mode="measured", stable_evolution=False, threshold_mode="median", **common)
    expected, _ = create_binary_labyrinth_pattern((64,64), stripe_width=8,
                                                  region='custom', k0=.7, eps=.8, **common)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('fraction', [0., .15, .35, .5, .8, 1.])
def test_requested_fraction_survives_soft_walls_and_smoothing(fraction):
    pattern, meta = create_domain_pattern((64,80), period=16, up_fraction=fraction,
                                         sigma=1.2, domain_conversion="soft", seed=8, use_gpu=False, n_steps=30)
    assert abs(np.mean(pattern > 0) - fraction) <= 1 / pattern.size
    assert np.isfinite(pattern).all() and np.max(np.abs(pattern)) <= 1
    assert meta['up_fraction_measured'] == np.mean(pattern > 0)
    assert meta['mean_mz'] == pattern.mean()


def test_target_mean_changes_evolved_phase_fraction_without_quantile():
    fractions=[]
    for mean in (-.5, 0., .5):
        pattern, _=create_domain_pattern((64,64), period=16, target_mean=mean,
                                        seed=4, n_steps=40, use_gpu=False)
        fractions.append(np.mean(pattern > 0))
    assert fractions[0] < fractions[1] < fractions[2]


def test_lattice_order_is_reproducible_and_changes_structure():
    random, _=create_domain_pattern((64,64),period=16,seed=4,n_steps=30)
    lattice, _=create_domain_pattern((64,64),period=16,seed=4,n_steps=30,lattice_order=1.)
    repeat, _=create_domain_pattern((64,64),period=16,seed=4,n_steps=30,lattice_order=1.)
    np.testing.assert_array_equal(lattice,repeat)
    assert np.sqrt(np.mean((random-lattice)**2)) > .1


def test_pipeline_converts_general_domain_physical_lengths():
    config=MagneticPatternConfig(pattern_type_method='domain_pattern',shape=(64,64),
        real_space_pixel_size=2e-9,pattern_config=dict(period=32e-9,sigma=2e-9,
        up_fraction=.3,seed=4,n_steps=30))
    pattern,meta=config.create_pattern()
    assert meta['period_requested_px'] == 16.
    assert abs(np.mean(pattern>0)-.3) < 1/pattern.size


@pytest.mark.parametrize('options', [dict(period=-1),dict(k0=0),dict(up_fraction=1.1),
    dict(lattice_order=1.1),dict(target_mean=np.nan),dict(n_steps=-1)])
def test_invalid_general_controls_fail(options):
    params=dict(period=16);params.update(options)
    with pytest.raises(ValueError):create_domain_pattern((32,32),**params)


def test_catalogue_handles_uniform_pattern():
    metrics=measure_domain_pattern(np.ones((32,32)))
    assert metrics['morphology']=='uniform'
    assert metrics['sixfold_order'] is None
    assert metrics['up_fraction']==1


def test_ordered_minority_domains_have_resolved_hexagonal_centers():
    pattern, _ = create_domain_pattern((128,128), period=16, lattice_order=1.,
        up_fraction=.15, sigma=.7, softness=.6, seed=11, use_gpu=False)
    metrics = measure_domain_pattern(pattern)
    assert metrics['bubble_count'] > 20
    assert metrics['sixfold_order'] > .5
    assert pattern.std() > .3


def test_seed_controls_driving_noise():
    args=dict(period=16, noise_amp=.02, seed=12, n_steps=30, use_gpu=False)
    first,_=create_domain_pattern((64,64),**args)
    second,_=create_domain_pattern((64,64),**args)
    np.testing.assert_array_equal(first,second)


def test_general_domains_work_in_saved_experiment(tmp_path):
    from scattering_calculator import simulation_pipelines as sim
    experiment=sim.ExperimentConfig(
        xray=sim.XRayConfig(energy=790.,photon_flux=1000.,pol='CR'),
        simulation=sim.SimulationConfig(shape=(32,40),real_space_pixel_size=4e-9),
        sample=sim.SampleConfig(recipe='Co(6)',max_slice_thickness=2e-9),
        magnetic_pattern=sim.MagneticPatternConfig(pattern_type_method='domain_pattern',
            pattern_config=dict(period=64e-9,up_fraction=.3,lattice_order=.5,seed=4,n_steps=30)),
        aperture=sim.FrontApertureConfig(aperture_method=None),
        illumination=sim.IlluminationConfig(illumination_function=None),
        propagation=sim.SamplePropagatorConfig(propagator_method='Scalar',propagator_config={'propagate':True}),
        detector=sim.DetectorConfig(shape=(10,12),detector_center=(5,6),artifacts_config={'sigma_photon':0}),
        outputs=sim.OutputConfig(path=tmp_path/'domains.h5'))
    results=sim.simulate_experiment(experiment)
    image=results.read('detector_ideal')
    assert image.shape==(10,12) and np.isfinite(image).all() and image.sum()>0


def test_measured_period_uses_rectangular_grid_frequencies():
    xx=np.arange(96)[None,:]
    field=np.broadcast_to(np.sin(2*np.pi*xx/16),(64,96))
    assert measure_domain_pattern(field)['dominant_period_px']==pytest.approx(16.)


@pytest.mark.parametrize("fraction", [.15, .3, .5, .8])
def test_default_fraction_changes_area_not_domain_amplitude(fraction):
    pattern,meta=create_domain_pattern((64,80),period=16,up_fraction=fraction,seed=11)
    assert set(np.unique(pattern)) == {-1., 1.}
    assert abs(np.mean(pattern == 1) - fraction) <= 1 / pattern.size
    assert pattern.mean() == pytest.approx(2 * np.mean(pattern == 1) - 1)
    assert meta['domain_conversion'] == 'hard'



def test_native_mode_matches_tutorial03_unrescaled_solver():
    from scattering_calculator.sample_generator.gray_scott_generator_binary import generate
    controls=dict(k0=.2, eps=.6, target_mean=0., n_steps=80, noise_amp=0., seed=7, use_gpu=False)
    _,binary,_,_=generate(batch=1,H=64,W=80,region='custom',**controls)
    pattern,meta=create_domain_pattern((64,80),period=16,rescale_mode='native',
        stable_evolution=False,**controls)
    np.testing.assert_array_equal(pattern,binary[0])
    assert meta['rescale_factor']==1.
    assert meta['generated_H']==64 and meta['generated_W']==80
