"""Binary domain generation, physical walls, sampling and saved hologram state."""
import json

import h5py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from scattering_calculator.simulation_pipelines.simulation_configuration import MagneticPatternConfig
from scattering_calculator.simulation_pipelines.magnetic_parameters import sample_constrained_domain_parameters
from scattering_calculator.simulation_pipelines.pipelines.hologram_pipeline import (
    HologramPipeline, HologramPipelineConfig, HologramPipelineRanges,
)


def test_constrained_walls_blur_native_binary_grid_in_physical_units():
    cfg = MagneticPatternConfig(pattern_type_method='constrained_domain_pattern',
        shape=(64, 80), real_space_pixel_size=2e-9, pattern_config=dict(
            characteristic_length=32e-9, sigma=4e-9, target_magnetization=.4,
            n_steps=20, max_length_iterations=1, seed=4))
    smooth, meta = cfg.create_pattern()
    assert set(np.unique(cfg.binary_pattern)) == {-1., 1.}
    np.testing.assert_allclose(smooth, gaussian_filter(cfg.binary_pattern, 2, mode='wrap'))
    assert np.any(np.abs(smooth) < .9)
    assert meta['wall_sigma_px'] == 2
    assert meta['wall_sigma_m'] == 4e-9
    assert abs(cfg.binary_pattern.mean() - .4) <= 1 / smooth.size
    assert smooth.mean() == pytest.approx(cfg.binary_pattern.mean())


@pytest.mark.parametrize('controls', [dict(characteristic_length=16e-9),
    dict(characteristic_length=64e-9, sigma=1e-9),
    dict(characteristic_length=64e-9, target_magnetization=.9)])
def test_explicit_grid_rejects_underresolved_domains_or_walls(controls):
    cfg = MagneticPatternConfig(pattern_type_method='constrained_domain_pattern',
        shape=(32, 32), real_space_pixel_size=2e-9, pattern_config=controls)
    with pytest.raises(ValueError, match='simulation pixels'):
        cfg.create_pattern()


def test_diverse_sampler_is_reproducible_without_requested_state():
    first = np.random.default_rng(5)
    second = np.random.default_rng(5)
    samples = [sample_constrained_domain_parameters(first) for _ in range(100)]
    assert samples == [sample_constrained_domain_parameters(second) for _ in range(100)]
    assert {s['initial_state'] for s in samples} == {'random', 'stripes', 'hexagonal', 'worms'}
    assert all(s['target_magnetization'] in (None, -1., 1.) for s in samples)
    assert {s['target_magnetization'] for s in samples} == {None, -1., 1.}
    assert all('requested_state' not in s and 'k0' not in s for s in samples)
    assert all(s['max_length_iterations'] == 1 for s in samples)


def small_hologram_config(**updates):
    values = dict(recipe='Au(10)/SiN(20)/Co(6)', detector_shape=(32, 32), detector_distance=.005,
        detector_center=(16, 16), oversampling=2, xray_photon_flux=100.,
        aperture_types=['OH', 'RH'], aperture_radii=[30e-9, 6e-9],
        aperture_centers=[(0., 0.), (80e-9, 0.)], aperture_sigmas=[1e-9, 1e-9],
        aperture_lengths=[0., 0.], aperture_angles=[0., 0.], aperture_ellipticities=[1., 1.],
        aperture_roughnesses=[0., 0.], aperture_roughness_modes=[(0, 0), (0, 0)],
        aperture_seeds=[1, 2], aperture_top_radius_factors=[1., 1.],
        illumination_function=None, beamstop_method=None, propagator_method='Scalar',
        artifacts_config={'sigma_photon': 0}, random_seed=4,
        pattern_config=dict(characteristic_length=64e-9, sigma=2e-9,
                            target_magnetization=.4, n_steps=10, max_length_iterations=1, seed=7))
    values.update(updates)
    return HologramPipelineConfig(**values)


@pytest.mark.parametrize('target', [.4, None])
def test_actual_hologram_saves_binary_state_walls_parameters_and_sampling(tmp_path, target):
    path = tmp_path / 'hologram.h5'
    cfg = small_hologram_config()
    cfg.pattern_config['target_magnetization'] = target
    pipeline = HologramPipeline(config=cfg, ranges=HologramPipelineRanges(),
                               output_path=path, n_samples=1, verbose=False)
    pipeline.run()
    with h5py.File(path) as h5:
        group = h5['00000']
        meta = group['metadata/sample/magnetic_pattern']
        controls = json.loads(meta['generation_json'][()].decode())
        assert controls['seed'] == 7
        assert controls['target_magnetization'] == target
        assert controls['magnetization_constrained'] == (target is not None)
        assert controls['n_steps_requested'] == 10
        assert controls['wall_sigma_px'] >= 2
        assert controls['characteristic_length_px'] >= 16
        assert controls['max_length_iterations'] == 1
        assert not controls['calibration_enabled']
        assert len(controls['calibration_history']) == 1
        assert 'trace' in controls and 'calibration_history' in controls
        assert 'binary_pattern' not in meta
        assert 'analysis_mask' not in meta
        classification = json.loads(meta['classification_json'][()].decode())
        assert classification['periodic']
        assert not classification['boundary_context_periodic']
        assert classification['stripe_count'] == meta['stripe_count'][()]
        assert classification['bubble_count'] == meta['bubble_count'][()]
        assert classification['state'] == meta['state'][()].decode()
        assert classification['stripe_count'] == classification['positive_stripe_count'] + classification['negative_stripe_count']
        assert group['metadata/sample/effective_oversampling'][()] > 2
        assert np.any((np.abs(group['magnetic_pattern_oh'][()]) > .01) &
                      (np.abs(group['magnetic_pattern_oh'][()]) < .99))
        for helicity in ('CR', 'CL'):
            assert np.isfinite(group[f'{helicity}/ideal'][()]).all()


def test_sampling_allocation_guard_fails_before_large_grid(tmp_path):
    pipeline = HologramPipeline(config=small_hologram_config(magnetic_pattern_max_grid_axis=32),
        ranges=HologramPipelineRanges(), output_path=tmp_path/'too_large.h5', n_samples=1, verbose=False)
    with h5py.File(pipeline.output_path, 'w') as h5:
        with pytest.raises(ValueError, match='magnetic_pattern_max_grid_axis'):
            pipeline._simulate_one(h5, 0)


@pytest.mark.parametrize('use_oh', [False, True])
def test_shared_experiment_hdf5_retains_constrained_provenance_and_unit_walls(tmp_path, use_oh):
    from scattering_calculator import simulation_pipelines as sim
    from scattering_calculator.simulation_pipelines.experiment import ScatteringExperiment
    config = sim.ExperimentConfig(
        xray=sim.XRayConfig(energy=790., photon_flux=100., pol='CR'),
        simulation=sim.SimulationConfig(shape=(32, 40), real_space_pixel_size=4e-9),
        sample=sim.SampleConfig(recipe='Co(6)', max_slice_thickness=2e-9),
        magnetic_pattern=sim.MagneticPatternConfig(pattern_type_method='constrained_domain_pattern',
            pattern_config=dict(characteristic_length=64e-9, sigma=8e-9,
                target_magnetization=.4, seed=7, n_steps=10, max_length_iterations=1)),
        aperture=sim.FrontApertureConfig(aperture_method=None),
        illumination=sim.IlluminationConfig(illumination_function=None),
        propagation=sim.SamplePropagatorConfig(propagator_method='Scalar', propagator_config={'propagate': True}),
        detector=sim.DetectorConfig(shape=(10, 12), detector_center=(5, 6), artifacts_config={'sigma_photon': 0}),
        outputs=sim.OutputConfig(path=tmp_path/'shared.h5'))
    if use_oh:
        config.aperture = sim.FrontApertureConfig(aperture_method='FTH_circular',
            aperture_config=dict(apertures_type=['OH'], apertures_radius=[20e-9],
                apertures_center=[(0., 0.)], apertures_depth=[6e-9],
                apertures_top_radius_factor=[1.]))
    run = ScatteringExperiment(config).setup()
    np.testing.assert_allclose(np.linalg.norm(run.sample.magnetization, axis=-1), 1.)
    assert np.any(run.sample.magnetization[..., 1] > .1)
    result = sim.simulate_experiment(config)
    with h5py.File(result.path) as h5:
        state = h5['runs/000000/magnetic_state']
        assert json.loads(state['parameters_json'][()])['sigma'] == 8e-9
        meta = json.loads(state['generation_json'][()])
        assert meta['wall_sigma_px'] == 2
        assert meta['seed'] == 7
        assert 'binary_pattern' not in state
        assert 'analysis_mask' not in state
        assert 'state' in state['classification']
        assert state['classification/periodic'][()]
        if use_oh:
            from scattering_calculator.sample_generator.domain_analysis import classify_magnetic_domains
            mask = run.config.aperture.create_supportmask(
                output_shape=run.config.magnetic_pattern.binary_pattern.shape,
                output_pixel_size=run.config.simulation.real_space_pixel_size,
                aperture_types=('OH',)) > 0
            assert np.any(mask) and not np.all(mask)
            expected = classify_magnetic_domains(run.config.magnetic_pattern.binary_pattern, analysis_mask=mask, periodic=True)
            assert state['classification/bubble_count'][()] == expected['bubble_count']
            assert state['classification/stripe_count'][()] == expected['stripe_count']


def test_generation_roi_uses_actual_oh_bounds_with_resolved_context():
    from scipy.fft import next_fast_len
    mask = np.zeros((2400, 2400), dtype=bool)
    mask[1048:1353, 1048:1353] = True
    shape, slices, _ = HologramPipeline._constrained_domain_roi(mask, 1., 275., 10.6)
    assert shape == (next_fast_len(305 + 2*180),) * 2
    assert shape[0] < 700  # Previously 1702 pixels for a comparable field.
    assert np.count_nonzero(mask[slices]) == np.count_nonzero(mask)
    assert slices[0].start <= 1048 - 180
    assert slices[0].stop >= 1353 + 180
    wider, _, _ = HologramPipeline._constrained_domain_roi(mask, 1., 275., 10.6, context_periods=2.)
    assert wider[0] > shape[0]


def test_generation_roi_clips_at_full_grid_and_covers_disjoint_ohs():
    mask = np.zeros((256, 384), dtype=bool)
    mask[:15, :20] = True
    mask[220:240, 300:325] = True
    shape, slices, _ = HologramPipeline._constrained_domain_roi(mask, 1., 32., 2.)
    assert shape[0] <= 256 and shape[1] <= 384
    assert np.count_nonzero(mask[slices]) == mask.sum()


def test_uniform_generation_roi_needs_no_domain_spacing_margin():
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[490:535, 490:535] = True
    shape, slices, _ = HologramPipeline._constrained_domain_roi(mask, 1., 300., 20., saturated=True)
    assert max(shape) < 50
    assert mask[slices].sum() == mask.sum()



def test_hologram_sampling_defaults_disable_spacing_calibration(tmp_path):
    pipeline = HologramPipeline(config=HologramPipelineConfig(recipe='Co(6)'),
        ranges=HologramPipelineRanges(), output_path=tmp_path/'defaults.h5', n_samples=1, verbose=False)
    assert pipeline._sample_params()['pattern_config']['max_length_iterations'] == 1


def test_optional_explicit_calibration_remains_available(tmp_path):
    pipeline = HologramPipeline(config=HologramPipelineConfig(recipe='Co(6)',
        pattern_config={'characteristic_length': 160e-9, 'max_length_iterations': 3}),
        ranges=HologramPipelineRanges(), output_path=tmp_path/'calibrated.h5', n_samples=1, verbose=False)
    assert pipeline._sample_params()['pattern_config']['max_length_iterations'] == 3



def test_sampler_supports_free_sampled_and_fixed_magnetization():
    free = sample_constrained_domain_parameters(np.random.default_rng(7))
    fixed = sample_constrained_domain_parameters(np.random.default_rng(7), target_magnetization=-.6)
    rng = np.random.default_rng(5)
    sampled = [sample_constrained_domain_parameters(rng, constrain_magnetization=True) for _ in range(100)]
    assert free['target_magnetization'] is None
    assert fixed['target_magnetization'] == -.6
    assert all(-1 <= s['target_magnetization'] <= 1 for s in sampled)
    assert any(abs(s['target_magnetization']) == 1 for s in sampled)


@pytest.mark.parametrize('factory', [False, True])
def test_detector_response_callbacks_survive_sweep_sampling(tmp_path, factory):
    def gain(energy_eV):
        raise AssertionError('Gain must not be called by parameter sampling')
    def response(linear_adu):
        raise AssertionError('Response must not be called by parameter sampling')
    callbacks = {'counts_per_photon': gain, 'nonlinear_response': response}
    override = (lambda params: callbacks) if factory else callbacks
    pipeline = HologramPipeline(config=small_hologram_config(),
        ranges=HologramPipelineRanges(detector_params=override),
        output_path=tmp_path/'callbacks.h5', n_samples=1, verbose=False)
    sampled = pipeline._sample_params()
    assert sampled['detector_params']['counts_per_photon'] is gain
    assert sampled['detector_params']['nonlinear_response'] is response
    default = HologramPipeline(config=small_hologram_config(), ranges=HologramPipelineRanges(),
        output_path=tmp_path/'linear.h5', n_samples=1, verbose=False)._sample_params()
    assert default['detector_params']['nonlinear_response'] is None


def test_nonlinear_sweep_response_and_energy_gain_are_applied_and_recorded(tmp_path):
    calls = []
    def gain(energy_eV):
        return .1 * energy_eV
    def compress(linear_adu):
        calls.append(linear_adu.copy())
        return .5 * linear_adu
    paths = []
    for name, response in [('linear', None), ('nonlinear', compress)]:
        cfg = small_hologram_config()
        cfg.detector_params = dict(readout_noise_average=0., readout_noise_sigma=0.,
            detector_threshold=1e12, quantum_efficiency=1., noise_seed=42)
        cfg.measurement_config = dict(number_frames=1, exposure_time=1., max_counts_per_image=None,
                                      acquisition_mode='accumulated')
        path = tmp_path/f'{name}.h5'
        HologramPipeline(config=cfg, ranges=HologramPipelineRanges(detector_params={
            'counts_per_photon': gain, 'nonlinear_response': response}),
            output_path=path, n_samples=1, verbose=False).run()
        paths.append(path)
    assert len(calls) == 2  # One physical response for each helicity.
    with h5py.File(paths[0]) as linear, h5py.File(paths[1]) as nonlinear:
        for pol in ('CR', 'CL'):
            np.testing.assert_allclose(nonlinear[f'00000/{pol}/detected'][()],
                                       .5*linear[f'00000/{pol}/detected'][()])
            meta = nonlinear['00000/metadata']
            energy = meta[f'detector_response/{pol}/energy_eV'][()]
            assert meta[f'detector_response/{pol}/counts_per_photon'][()] == pytest.approx(.1*energy)
        assert linear['00000/metadata/detector_response/model'].asstr()[()] == 'linear'
        assert nonlinear['00000/metadata/detector_response/model'].asstr()[()] == 'nonlinear'
        assert 'compress' in nonlinear['00000/metadata/detector_params/nonlinear_response'].asstr()[()]
        assert 'gain' in nonlinear['00000/metadata/detector_params/counts_per_photon'].asstr()[()]


def test_verbose_pipeline_reports_magnetic_details_and_saves_timings(tmp_path, capsys):
    path = tmp_path / 'timed.h5'
    HologramPipeline(config=small_hologram_config(), ranges=HologramPipelineRanges(),
        output_path=path, n_samples=1, verbose=True, detailed_logging=True).run()
    output = capsys.readouterr().out
    assert 'RMS update=' not in output
    assert 'relaxation 1/' not in output
    for label in ('magnetic grid=', 'pitch=', 'attempt 1/1',
                  'relaxation breakdown', 'spectral_update=', 'area_projection=',
                  'wall smoothing:', '3D magnetization mapping', 'magnetic diagnostics'):
        assert label in output
    with h5py.File(path) as h5:
        timing = h5['00000/metadata/timing_seconds']
        assert timing['domain_relaxation_and_walls'][()] >= 0
        assert timing['3D_magnetization_mapping'][()] >= 0
        generation = json.loads(h5['00000/metadata/sample/magnetic_pattern/generation_json'][()])
        assert generation['timings_seconds']['wall_smoothing'] >= 0
        assert not generation['component_diagnostics']
        assert 'state' in json.loads(h5['00000/metadata/sample/magnetic_pattern/classification_json'][()])


def test_saturated_sampler_generates_uniform_states_with_both_polarities():
    from scattering_calculator.sample_generator.domain_analysis import classify_magnetic_domains
    rng = np.random.default_rng(42)
    targets = set()
    for _ in range(10):
        controls = sample_constrained_domain_parameters(rng, saturation_probability=1.)
        target = controls['target_magnetization']
        targets.add(target)
        cfg = MagneticPatternConfig(pattern_type_method='constrained_domain_pattern',
            shape=(32, 32), real_space_pixel_size=4e-9,
            pattern_config={**controls, 'characteristic_length': 128e-9, 'sigma': 8e-9})
        smooth, _ = cfg.create_pattern()
        np.testing.assert_allclose(smooth, target)
        state = classify_magnetic_domains(cfg.binary_pattern, periodic=True)
        assert (state['state'], state['bubble_count'], state['stripe_count']) == ('saturated', 0, 1)
    assert targets == {-1., 1.}
    assert sample_constrained_domain_parameters(rng, saturation_probability=0.)['target_magnetization'] is None
    assert sample_constrained_domain_parameters(rng, saturation_probability=1., target_magnetization=.4)['target_magnetization'] == .4


def test_sweep_sampler_keeps_weak_alignment_and_a_broad_order_range():
    rng = np.random.default_rng(123)
    samples = [sample_constrained_domain_parameters(rng) for _ in range(3000)]
    lengths = np.array([sample['characteristic_length'] for sample in samples])
    assert np.all((lengths >= 64e-9) & (lengths <= 1000e-9))
    # Equal log intervals, including fine domains, receive equal draw probability.
    assert .46 < np.mean(lengths < np.sqrt(64e-9 * 1000e-9)) < .54
    smooth = [s for s in samples if s['disorder'] == 0.]
    assert .46 < len(smooth) / len(samples) < .54
    assert all(s['alignment'] == s['initial_order'] == 0. and
               s['initial_state'] == 'random' and s['wall_stiffness'] == 2. and
               s['n_steps'] == 400 and s['time_step'] == .2 for s in smooth)
    diverse = [s for s in samples if s['disorder'] != 0.]
    alignment = np.array([s['alignment'] for s in diverse])
    assert np.all((alignment >= 0) & (alignment < 24))
    assert .44 < np.mean(alignment == 0) < .56
    assert .19 < alignment.mean() < .31
    assert all(0 <= s['disorder'] <= .8 for s in diverse)
    assert .20 < np.mean([s['initial_order'] > .75 for s in diverse]) < .30
    assert .20 < np.mean([s['initial_order'] < .25 for s in diverse]) < .30


def test_sweep_magnetization_mixture_has_free_gaussian_and_saturated_draws():
    from scattering_calculator.simulation_pipelines.magnetic_parameters import sample_sweep_domain_parameters
    rng = np.random.default_rng(31)
    targets = [sample_sweep_domain_parameters(rng, i)['target_magnetization'] for i in range(6000)]
    free = sum(target is None for target in targets)
    saturated = [target for target in targets if target is not None and abs(target) == 1]
    gaussian = np.array([target for target in targets if target is not None and abs(target) != 1])
    assert .45 < free/len(targets) < .51
    assert .45 < len(gaussian)/len(targets) < .51
    assert .03 < len(saturated)/len(targets) < .05
    assert set(saturated) == {-1., 1.}
    assert np.all(np.abs(gaussian) <= .9)
    assert abs(gaussian.mean()) < .02
    assert .19 < gaussian.std() < .23
    with pytest.raises(ValueError, match='sum to one'):
        sample_sweep_domain_parameters(rng, free_probability=.6)


def test_default_progress_omits_stage_details_but_saves_timings(tmp_path, capsys):
    path = tmp_path/'concise.h5'
    HologramPipeline(config=small_hologram_config(), ranges=HologramPipelineRanges(),
        output_path=path, n_samples=1, verbose=True).run()
    output = capsys.readouterr().out
    assert 'Simulating...' in output and 'Total:' in output and 'Pipeline complete' in output
    for label in ['Experiment setup:', 'Magnetic pattern:', 'Aperture construction:',
                  'Exit-wave propagation (CR + CL):', 'Shared prefix:', 'CR suffix:', 'CL suffix:',
                  'final FFTs:', 'Detector acquisition (CR + CL):', 'Save results:']:
        assert label in output
    for text in ['relaxation breakdown', 'detector noise:', 'magnetic grid=', 'timing:', 'generator total']:
        assert text not in output
    with h5py.File(path) as h5:
        assert h5['00000/metadata/timing_seconds/domain_relaxation_and_walls'][()] >= 0


@pytest.mark.parametrize('compression', ['lzf', 'gzip', None])
def test_lossless_output_compression_is_configurable(tmp_path, compression):
    path = tmp_path/'compression.h5'
    HologramPipeline(config=small_hologram_config(hdf5_compression=compression),
        ranges=HologramPipelineRanges(), output_path=path, n_samples=1, verbose=False).run()
    with h5py.File(path) as h5:
        for name in ['CR/exit_wave', 'CL/ideal', 'CR/detected', 'supportmask', 'magnetic_pattern_oh']:
            assert h5[f'00000/{name}'].compression == compression
            assert np.isfinite(h5[f'00000/{name}'][()]).all()


def test_compression_modes_roundtrip_identical_arrays_and_dtypes(tmp_path):
    arrays = None
    for compression in ['lzf', 'gzip', None]:
        path = tmp_path/f'{compression}.h5'
        config = small_hologram_config(hdf5_compression=compression,
            detector_params={'noise_seed': 7, 'counts_per_photon': 100.,
                             'readout_noise_average': 0., 'readout_noise_sigma': 0.})
        HologramPipeline(config=config, ranges=HologramPipelineRanges(),
            output_path=path, n_samples=1, verbose=False).run()
        with h5py.File(path) as h5:
            names = ['CR/exit_wave', 'CL/exit_wave', 'CR/ideal', 'CL/ideal',
                     'CR/detected', 'CL/detected', 'supportmask', 'magnetic_pattern_oh']
            current = {name: h5[f'00000/{name}'][()] for name in names}
            assert current['CR/exit_wave'].dtype == np.complex64
            assert current['CR/ideal'].dtype == np.float32
            if arrays is None:
                arrays = current
            else:
                for name in names:
                    assert current[name].dtype == arrays[name].dtype
                    np.testing.assert_array_equal(current[name], arrays[name])


@pytest.mark.parametrize('center,flat', [((31.2, 42.7), False), ((-2., 90.), False), ((31.2, 42.7), True)])
def test_compatible_distance_uses_exact_detector_sampling(center, flat):
    from scattering_calculator.simulation_pipelines.magnetic_parameters import compatible_detector_distance
    from scattering_calculator.simulation_pipelines.simulation_configuration import DetectorConfig, XRayConfig
    from scattering_calculator.utils.physics import photon_energy_wavelength
    beam = XRayConfig(energy=780., photon_flux=1.)
    beam.setup()
    def resolution(distance):
        detector = DetectorConfig(shape=(64, 80), pixel_size=13.5e-6,
            detector_center=center, sample_to_detector_distance=distance,
            ignore_flat_detector_curvature=flat)
        detector.setup()
        return detector.calc_realspace_resolution(beam.beam_params)
    target = resolution(.1)/4
    settings = dict(wavelength=photon_energy_wavelength(780.), detector_shape=(64, 80),
        pixel_size=13.5e-6, center=center, target_pitch=target,
        max_oversampling=4, ignore_curvature=flat)
    for _ in range(10):
        actual = compatible_detector_distance(.4, .05, **settings)
        assert .05 <= actual <= .1*(1+1e-10)
        assert resolution(actual)/4 <= target
    assert compatible_detector_distance(.05, .05, **settings) == .05
    assert compatible_detector_distance(.4, .2, **settings) is None


def test_distance_adjustment_precedes_aperture_sampling_and_preserves_texture():
    controls = dict(characteristic_length=160e-9, sigma=5e-9, seed=3,
                    target_magnetization=.2, n_steps=20, max_length_iterations=1)
    cfg = small_hologram_config(magnetic_pattern_distance_min=.005,
                               magnetic_pattern_max_grid_axis=512)
    seen = []
    pipeline = HologramPipeline(config=cfg, ranges=HologramPipelineRanges(
        detector_distance=lambda _: .4, pattern_config=lambda _: controls.copy(),
        aperture_config=lambda p: seen.append(p['detector_distance']) or {}),
        output_path='/tmp/unused-compatible-distance.h5', n_samples=1, verbose=False)
    params = pipeline._sample_params()
    assert .005 <= params['detector_distance'] < .4
    assert params['detector_distance_original'] == .4
    assert seen == [params['detector_distance']]
    assert params['pattern_config'] == controls


def test_conditional_distance_samples_full_feasible_interval(monkeypatch):
    from scattering_calculator.simulation_pipelines.magnetic_parameters import compatible_detector_distance
    intervals = []
    def midpoint(low, high):
        intervals.append((low, high))
        return (low+high)/2
    monkeypatch.setattr(np.random, 'uniform', midpoint)
    settings = dict(wavelength=1.6e-9, detector_shape=(256, 256), pixel_size=20e-6,
                    center=(128, 128), max_oversampling=16, sample_feasible=True)
    # With broad domains, use the entire requested interval even though 40 cm fits.
    distance = compatible_detector_distance(.4, .04, target_pitch=1e-6, **settings)
    assert intervals[-1] == (.04, .4)
    assert distance == pytest.approx(.22)
    # Narrow walls shrink the upper bound without altering their requested pitch.
    fine = compatible_detector_distance(.4, .04, target_pitch=2e-9, **settings)
    assert intervals[-1][0] == .04
    assert .04 < intervals[-1][1] < .4
    assert .04 < fine < distance
