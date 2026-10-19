"""Constraints, ordering, scale calibration and workflow wiring for dipolar domains."""
import numpy as np
import pytest
from scattering_calculator.sample_generator import create_constrained_domain_pattern as generate
from scattering_calculator.simulation_pipelines import simulation_configuration as sim


@pytest.mark.parametrize('magnetization', [-1.,-.6,0.,.35,1.])
def test_fixed_binary_area_is_enforced_during_relaxation(magnetization):
    pattern,meta=generate((48,64),12,target_magnetization=magnetization,seed=9,n_steps=80)
    count=round((magnetization+1)*.5*pattern.size)
    assert set(np.unique(pattern)) <= {-1.,1.}
    assert np.count_nonzero(pattern==1)==count
    assert abs(pattern.mean()-magnetization) <= 1/pattern.size
    assert all(record['up_pixels']==count for record in meta['trace'])
    if abs(magnetization)==1:
        assert meta['measured_length_px'] is None
        assert meta['steps_completed']==0


def test_reproducible_pin_disorder_and_constraints():
    args=dict(target_magnetization=.4,seed=4,n_steps=80,disorder=.3)
    a,ma=generate((48,64),12,**args)
    b,mb=generate((48,64),12,**args)
    np.testing.assert_array_equal(a,b)
    assert ma['trace']==mb['trace']


def test_alignment_orders_stripes_and_direction_rotates_them():
    args=dict(seed=7,n_steps=250,max_length_iterations=1)
    _,isotropic=generate((96,96),16,**args)
    horizontal,aligned=generate((96,96),16,alignment=8,**args)
    vertical,rotated=generate((96,96),16,alignment=8,direction=np.pi/2,**args)
    assert aligned['nematic_order'] > .8
    assert aligned['nematic_order'] > isotropic['nematic_order']+.4
    assert rotated['nematic_order'] > .8
    assert np.mean(horizontal[1:] != horizontal[:-1]) > np.mean(horizontal[:,1:] != horizontal[:,:-1])
    assert np.mean(vertical[:,1:] != vertical[:,:-1]) > np.mean(vertical[1:] != vertical[:-1])


def test_length_calibration_records_measured_outcome():
    _,meta=generate((128,128),16,seed=7,n_steps=250)
    assert len(meta['calibration_history'])<=5
    assert meta['length_target_met']
    assert meta['length_relative_error'] <= .15
    assert meta['measured_length_px'] == meta['measurements']['radial_period_px']


def test_ordered_stripe_seed_retains_exact_commensurate_period():
    _,meta=generate((64,64),16,initial_state='stripes',initial_order=1.,alignment=8,seed=7,n_steps=100)
    assert meta['measured_length_px']==pytest.approx(16,rel=.02)
    assert meta['nematic_order']>.99


def test_pipeline_converts_lengths_to_pixels():
    config=sim.MagneticPatternConfig(pattern_type_method='constrained_domain_pattern',
        shape=(48,64),real_space_pixel_size=2e-9,pattern_config=dict(
            characteristic_length=24e-9,disorder_length=4e-9,
            target_magnetization=.4,seed=4,n_steps=40))
    pattern,meta=config.create_pattern()
    assert meta['characteristic_length']==12.
    assert meta['disorder_length']==2.
    assert abs(pattern.mean()-.4) <= 1/pattern.size


@pytest.mark.parametrize('control', [dict(target_magnetization=1.1),dict(alignment=-1),
    dict(disorder=-1),dict(initial_order=2),dict(n_steps=0),dict(time_step=0),
    dict(initial_state='unknown'),dict(max_length_iterations=0)])
def test_invalid_controls(control):
    with pytest.raises(ValueError):generate((32,32),12,**control)


def test_constrained_domains_run_through_saved_experiment(tmp_path):
    from scattering_calculator import simulation_pipelines as workflow
    experiment=workflow.ExperimentConfig(
        xray=workflow.XRayConfig(energy=790.,photon_flux=1000.,pol='CR'),
        simulation=workflow.SimulationConfig(shape=(32,40),real_space_pixel_size=4e-9),
        sample=workflow.SampleConfig(recipe='Co(6)',max_slice_thickness=2e-9),
        magnetic_pattern=workflow.MagneticPatternConfig(pattern_type_method='constrained_domain_pattern',
            pattern_config=dict(characteristic_length=48e-9,target_magnetization=.4,seed=4,n_steps=50)),
        aperture=workflow.FrontApertureConfig(aperture_method=None),
        illumination=workflow.IlluminationConfig(illumination_function=None),
        propagation=workflow.SamplePropagatorConfig(propagator_method='Scalar',propagator_config={'propagate':True}),
        detector=workflow.DetectorConfig(shape=(10,12),detector_center=(5,6),artifacts_config={'sigma_photon':0}),
        outputs=workflow.OutputConfig(path=tmp_path/'constrained.h5'))
    results=workflow.simulate_experiment(experiment)
    ideal=results.read('detector_ideal')
    assert ideal.shape==(10,12) and np.isfinite(ideal).all() and ideal.sum()>0


def test_unresolved_length_target_is_reported_without_false_success():
    _,meta=generate((32,32),100,seed=3,n_steps=30,max_length_iterations=1)
    assert not meta['length_target_met']
    # Field convergence and satisfying an impossible size request are distinct.
    if meta['converged']:
        assert meta['residual'] < meta['convergence_tolerance']
    assert meta['length_relative_error'] > .15


@pytest.mark.parametrize('magnetization',[-.9,-.8,.8,.9])
@pytest.mark.parametrize('count',[1,2])
def test_finite_minority_worms_at_strong_net_magnetization(magnetization,count):
    from scipy.ndimage import label
    preferred_length = 32 if count == 1 else (16 if abs(magnetization) > .85 else 24)
    pattern,meta=generate((128,128),preferred_length,
        target_magnetization=magnetization,initial_state='worms',initial_order=1.,
        initial_domain_count=count,alignment=0.,n_steps=400,max_length_iterations=1,seed=7)
    minority=pattern<0 if magnetization>0 else pattern>0
    _,components=label(minority,structure=np.ones((3,3)))
    assert components==count
    assert not any(np.any(edge) for edge in (minority[0],minority[-1],minority[:,0],minority[:,-1]))
    assert meta['measurements']['mean_eccentricity']>.9
    assert abs(pattern.mean()-magnetization)<=1/pattern.size
    assert all(item['up_pixels']==meta['target_up_pixels'] for item in meta['trace'])


def test_worm_preparation_lengths_are_shared_physical_config_controls():
    config=sim.MagneticPatternConfig(pattern_type_method='constrained_domain_pattern',
        shape=(128,128),real_space_pixel_size=2e-9,pattern_config=dict(
            characteristic_length=48e-9,target_magnetization=.8,
            initial_state='worms',initial_order=1.,initial_domain_count=2,
            worm_length=150e-9,worm_waviness=16e-9,
            seed=7,n_steps=20,max_length_iterations=1))
    _,metadata=config.create_pattern()
    assert metadata['initial_worm_length_px']==pytest.approx(75.)
    assert metadata['initial_worm_waviness_px']==pytest.approx(8.)
    assert metadata['initial_domain_count']==2
