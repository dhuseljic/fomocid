"""Acceptance checks for single/scanned experiments, HDF5 reload and plotting."""
import json
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
import pytest
from scattering_calculator import simulation_pipelines as sim


def config_for(path):
    """Declare a small physical experiment with deterministic acquisition.

    Parameters
    ----------
    path : Path
        HDF5 destination.

    Returns
    -------
    ExperimentConfig
        Small Co specimen with a real Jones interaction and detector.
    """
    return sim.ExperimentConfig(
        xray=sim.XRayConfig(energy=790.,photon_flux=1000.,pol='LH',linear_polarization_angle=.2),
        simulation=sim.SimulationConfig(shape=(16,20),real_space_pixel_size=4e-9),
        sample=sim.SampleConfig(recipe='Co(6)',max_slice_thickness=2e-9),
        magnetic_pattern=sim.MagneticPatternConfig(pattern_type_method='magnon_wave',pattern_config={'period':40e-9,'amplitude':.1,'angle':.3}),
        aperture=sim.FrontApertureConfig(aperture_method=None),
        illumination=sim.IlluminationConfig(illumination_function=None),
        propagation=sim.SamplePropagatorConfig(propagator_method='Jones',propagator_config={'propagate':True}),
        detector=sim.DetectorConfig(shape=(10,12),detector_center=(5,6),sample_to_detector_distance=.03,
            detector_params={'noise_seed':10,'counts_per_photon':2.,'quantum_efficiency':.8,'readout_noise_average':4.,'readout_noise_sigma':2.,'detector_threshold':1e6},
            measurement_config={'exposure_time':.4,'number_frames':2,'max_counts_per_image':None},artifacts_config={'sigma_photon':0.}),
        beamstop=sim.BeamstopConfig(bs_method=None),
        outputs=sim.OutputConfig(path=path),
    )


def test_single_energy_uses_source_defaults_and_matches_the_standard_engine(tmp_path):
    """Save/reload a single exposure without a spectrum or duplicate detector.

    Parameters
    ----------
    tmp_path : Path
        Test output directory.

    Returns
    -------
    None
        Asserts persisted arrays, source settings and selected products.
    """
    c=config_for(tmp_path/'single.h5')
    results=sim.simulate_experiment(c)
    assert results.energies_eV==(790.,) and results.polarizations==('LH',)
    direct=sim.ScatteringExperiment(c).setup();direct.propagate()
    np.testing.assert_allclose(results.read('exit_wave'),direct.wavefront.exit_wave)
    np.testing.assert_allclose(results.read('detector_ideal'),direct.detect())
    np.testing.assert_allclose(results.read('fft_intensity').sum(),np.sum(abs(direct.wavefront.exit_wave)**2))
    assert results.config['detector']['detector_params']['counts_per_photon']==2.
    with h5py.File(results.path,'r') as file:
        assert file.attrs['status']=='complete'
        assert set(file['runs'])=={'000000'}
        assert 'sample_mask' not in file['runs/000000']
        assert file['runs/000000/exit_wave'].dtype.kind=='c'
    assert c.sample.sample_shape is None


def test_energy_and_polarization_scan_refreshes_optics_and_retains_detector_q(tmp_path):
    """A Cartesian scan changes material data and q without changing the specimen.

    Parameters
    ----------
    tmp_path : Path
        Test output directory.

    Returns
    -------
    None
        Asserts scan ordering, optical updates and field differences.
    """
    c=config_for(tmp_path/'scan.h5').with_changes(
        illumination={'energies_eV':(778.,790.),'polarizations':('CR','CL')},
        outputs={'save':('exit_wave','detector_ideal','magnetization')})
    r=sim.simulate_experiment(c)
    assert r.stack('detector_ideal').shape==(2,2,10,12)
    np.testing.assert_array_equal(r.read('magnetization'),r.read('magnetization',energy_index=1,polarization_index=1))
    assert not np.allclose(r.read('coordinates/qx'),r.read('coordinates/qx',energy_index=1))
    assert not np.array_equal(r.read('exit_wave'),r.read('exit_wave',polarization_index=1))
    with h5py.File(r.path,'r') as file:
        assert len(file['runs'])==4
        assert file['runs/000003'].attrs['polarization']=='CL'
        assert not np.array_equal(file['runs/000000/materials/refractive_indices'][()],file['runs/000002/materials/refractive_indices'][()])


def test_noise_is_repeatable_and_does_not_change_global_random_state(tmp_path):
    """A seeded acquisition reproduces readout noise as well as photon sampling.

    Parameters
    ----------
    tmp_path : Path
        Test output directory.

    Returns
    -------
    None
        Asserts repeatability and isolation from the caller's RNG state.
    """
    c=config_for(tmp_path/'a.h5')
    previous=np.random.get_state()
    a=sim.simulate_experiment(c).read('detector_measured')
    following=np.random.get_state()
    np.testing.assert_array_equal(previous[1],following[1]);assert previous[2:]==following[2:]
    b=sim.simulate_experiment(c.with_changes(outputs={'path':tmp_path/'b.h5'})).read('detector_measured')
    np.testing.assert_array_equal(a,b)


def test_existing_files_and_failed_scans_are_preserved(tmp_path):
    """Incomplete writes cannot replace a completed result, even with overwrite set.

    Parameters
    ----------
    tmp_path : Path
        Test output directory.

    Returns
    -------
    None
        Asserts exclusive publication and cleanup after a propagation failure.
    """
    path=tmp_path/'result.h5';c=config_for(path)
    sim.simulate_experiment(c);old=path.read_bytes()
    with pytest.raises(FileExistsError):sim.simulate_experiment(c)
    with patch.object(sim.ScatteringExperiment,'propagate',side_effect=RuntimeError('failed physics')):
        with pytest.raises(RuntimeError):sim.simulate_experiment(c.with_changes(outputs={'overwrite':True}))
    assert path.read_bytes()==old
    assert not list(tmp_path.glob('*.tmp'))


def test_reload_plot_and_all_in_one_do_not_resimulate_for_plotting(tmp_path):
    """Saved results can be plotted in a fresh session without a live experiment.

    Parameters
    ----------
    tmp_path : Path
        Test output directory.

    Returns
    -------
    None
        Asserts persistence, independent plotting and the combined convenience call.
    """
    import matplotlib.pyplot as plt
    r,fig=sim.run_experiment(config_for(tmp_path/'plot.h5'))
    assert (tmp_path/'plot.png').is_file();plt.close(fig)
    with patch.object(sim.ScatteringExperiment,'setup',side_effect=AssertionError('plot resimulated')):
        loaded=sim.load_results(r.path)
        fig=sim.plot_results(loaded,outputs=('detector_ideal',),save_path=tmp_path/'reloaded.png')
    assert (tmp_path/'reloaded.png').is_file();plt.close(fig)
    with pytest.raises(KeyError):loaded.read('sample_mask')
    with pytest.raises(IndexError):loaded.read('exit_wave',energy_index=2)


def test_legacy_scan_alias_is_checked_and_invalid_axes_fail_before_writing(tmp_path):
    """Legacy top-level energies are accepted without silently overriding illumination.

    Parameters
    ----------
    tmp_path : Path
        Test output directory.

    Returns
    -------
    None
        Asserts default resolution, alias conflicts and invalid scans.
    """
    c=config_for(tmp_path/'bad.h5')
    assert c.with_changes(energies_eV=(778.,)).scan_axes()[0]==(778.,)
    for update in [dict(illumination={'energies_eV':()}),dict(illumination={'energies_eV':(float('nan'),)}),
        dict(illumination={'polarizations':('invalid',)}),
        dict(energies_eV=(778.,),illumination={'energies_eV':(790.,)})]:
        with pytest.raises(ValueError):sim.simulate_experiment(c.with_changes(**update))
    assert not (tmp_path/'bad.h5').exists()


def test_mixed_stokes_outputs_keep_separate_coherent_modes(tmp_path):
    """Mixed polarization saves phase-bearing modes without inventing a single field.

    Parameters
    ----------
    tmp_path : Path
        Test output directory.

    Returns
    -------
    None
        Asserts incoherent field power agrees with the physical Stokes intensity.
    """
    c=config_for(tmp_path/'mixed.h5').with_changes(
        propagation={'propagator_method':'Stokes','propagator_config':{'propagate':True,'input_stokes':[1.,0.,0.,.4]}},
        outputs={'save':('exit_wave','exit_stokes','fft_intensity','detector_ideal')})
    r=sim.simulate_experiment(c)
    assert r.read('exit_wave').shape==(16,20,2,2)
    np.testing.assert_allclose(np.sum(abs(r.read('exit_wave'))**2,axis=(2,3)),r.read('exit_stokes')[...,0])


def test_detector_derived_scan_keeps_one_physical_grid_and_records_custom_inputs(tmp_path):
    """Energy changes cannot resize a specimen when the initial grid comes from a camera.

    Parameters
    ----------
    tmp_path : Path
        Test output directory.

    Returns
    -------
    None
        Asserts constant spatial sampling and persistence of externally supplied data.
    """
    c=config_for(tmp_path/'derived.h5').with_changes(
        simulation={'other_config':{'grid_mode':'detector','oversampling':1}},
        illumination={'energies_eV':(778.,790.)},outputs={'save':('exit_wave','detector_ideal')})
    m=np.zeros((10,12,3));m[...,0]=1.
    r=sim.simulate_experiment(c,magnetization=m)
    np.testing.assert_array_equal(r.read('coordinates/sample_x'),r.read('coordinates/sample_x',energy_index=1))
    assert r.config['simulation']['other_config']['sampling_reference_energy_eV']==790.
    assert c.simulation.other_config['grid_mode']=='detector'
    with h5py.File(r.path,'r') as file:np.testing.assert_array_equal(file['inputs/magnetization'][()],m)


def test_photon_kernels_and_persistent_defects_are_seeded_and_recorded(tmp_path):
    """A seed reproduces the complete sensor model, including event kernels.

    Parameters
    ----------
    tmp_path : Path
        Test output directory.

    Returns
    -------
    None
        Asserts stable persistent sensor fields and equivalent acquired images.
    """
    c=config_for(tmp_path/'sensor_a.h5').with_changes(
        detector={'artifacts_config':{'sigma_photon':1.,'photon_kernel_size':3,'photon_n_variants':2,
            'photon_n_classes':1,'average_hot_pixels':2.,'average_cold_pixels':1.,'cosmic_rays_per_second':.5}})
    a=sim.simulate_experiment(c)
    b=sim.simulate_experiment(c.with_changes(outputs={'path':tmp_path/'sensor_b.h5'}))
    np.testing.assert_array_equal(a.read('detector_measured'),b.read('detector_measured'))
    for key in ('camera_seed','photon_kernel_seed','photon_class_seed'):
        assert isinstance(a.config['detector']['artifacts_config'][key],int)
    assert 'camera_seed' not in c.detector.artifacts_config


def test_recipe_distance_method_lists_reuse_only_downstream_stages(tmp_path):
    """Expand sample/model/camera alternatives and reuse each physical exit field."""
    c=config_for(tmp_path/'parameters.h5')
    c.sample.recipe=['Co(4)','Co(6)']
    c.propagation.propagator_method=['Jones','Scalar']
    c.detector.sample_to_detector_distance=[.02,.04]
    original=sim.ScatteringExperiment.propagate
    calls=[]
    def counted(run):
        calls.append(run.config.sample.recipe)
        return original(run)
    with patch.object(sim.ScatteringExperiment,'propagate',counted):
        r=sim.simulate_experiment(c)
    assert len(calls)==4
    assert len(r.parameter_axes)==3
    assert np.array_equal(r.read('exit_wave',scan_index=0),r.read('exit_wave',scan_index=1))
    assert not np.allclose(r.read('coordinates/qx',scan_index=0),r.read('coordinates/qx',scan_index=1))
    with h5py.File(r.path) as f:
        assert len(f['runs'])==8
        assert f['runs/000001'].attrs['sample_propagation_reused']


def test_projection_energy_is_coordinate_only_and_source_lists_work(tmp_path):
    """Freeze detector q coordinates without freezing material/sample physics."""
    c=config_for(tmp_path/'projection.h5')
    c.xray=sim.XRayConfig(energy=[778.,790.],photon_flux=1000.,pol=['LH','LV'])
    c.detector.projection_energy=780.
    r=sim.simulate_experiment(c)
    assert r.energies_eV==(778.,790.) and r.polarizations==('LH','LV')
    assert np.array_equal(r.read('coordinates/qx'),r.read('coordinates/qx',energy_index=1))
    assert not np.array_equal(r.read('exit_wave'),r.read('exit_wave',energy_index=1))
    assert not np.array_equal(r.read('materials/refractive_indices'),r.read('materials/refractive_indices',energy_index=1))


def test_detector_lists_can_be_declared_in_constructor(tmp_path):
    """Constructor list alternatives resolve before component validation/setup."""
    c=config_for(tmp_path/'constructor.h5')
    c.detector=sim.DetectorConfig(shape=(8,8),detector_center=(4,4),
        sample_to_detector_distance=[.02,.04],analyzer_angle=[0.,np.pi/2],
        projection_energy=[778.,790.])
    c.outputs.save=('exit_wave','detector_ideal')
    r=sim.simulate_experiment(c)
    assert len(r.parameter_axes)==3
    assert np.array_equal(r.read('exit_wave'),r.read('exit_wave',scan_index=7))


def test_finite_distance_projection_override_keeps_physical_wavelength(tmp_path):
    """Coordinate energy cannot alter finite-distance diffraction physics."""
    c=config_for(tmp_path/'finite.h5')
    c.detector.detector_propagation_method='rayleigh_sommerfeld'
    c.detector.projection_energy=[778.,800.]
    c.outputs.save=('exit_wave','detector_ideal')
    r=sim.simulate_experiment(c)
    assert np.array_equal(r.read('detector_ideal'),r.read('detector_ideal',scan_index=1))
    assert not np.array_equal(r.read('coordinates/qx'),r.read('coordinates/qx',scan_index=1))


def test_harmonics_sum_projected_intensities_and_acquire_once(tmp_path):
    """Independent colors get physical optical data and sum on camera pixels."""
    c=config_for(tmp_path/'harmonics.h5')
    c.xray.energy=780.
    c.detector.detector_params.update(readout_noise_average=5.,readout_noise_sigma=0.)
    c.illumination.spectral_components=[
        sim.SpectralComponentConfig(energy=780.,weight=.9),
        sim.SpectralComponentConfig(energy_factor=2.,weight=.1)]
    c.outputs.save=('exit_wave','fft_intensity','detector_ideal','detector_measured')
    acquisition=sim.DetectorConfig.return_detected_hologram
    acquired=[]
    def once(camera):
        acquired.append(1)
        return acquisition(camera)
    with patch.object(sim.DetectorConfig,'return_detected_hologram',once):
        r=sim.simulate_experiment(c)
    assert len(acquired)==1
    components=[]
    for energy,fraction in [(780.,.9),(1560.,.1)]:
        separate=c.with_changes(xray={'energy':energy,'photon_flux':1000.*fraction},
            illumination={'spectral_components':None})
        components.append(sim.ScatteringExperiment(separate).run())
    np.testing.assert_allclose(r.read('detector_ideal'),sum(components),rtol=1e-12)
    with h5py.File(r.path) as f:
        parts=f['runs/000000/spectral_components']
        assert parts['000001'].attrs['energy_eV']==1560.
        assert not np.array_equal(parts['000000/qx'][()],parts['000001/qx'][()])
        np.testing.assert_allclose(parts['000000/detector_ideal'][()],components[0])
        np.testing.assert_allclose(parts['000001/detector_ideal'][()],components[1])
    assert r.read('exit_wave').shape==(16,20,2,2)
    assert r.parameter_axes=={}


def test_multicolor_detector_scan_reuses_both_colors(tmp_path):
    """Camera changes reuse each spectral exit wave and preserve component images."""
    c=config_for(tmp_path/'color_scan.h5')
    c.illumination.spectral_components=(sim.SpectralComponentConfig(weight=9.),
        sim.SpectralComponentConfig(energy_factor=2.,weight=1.))
    c.detector.sample_to_detector_distance=[.02,.04]
    c.outputs.save=('exit_wave','detector_ideal')
    r=sim.simulate_experiment(c)
    np.testing.assert_array_equal(r.read('exit_wave'),r.read('exit_wave',scan_index=1))
    for scan_index in range(2):
        expected=sum(r.read(f'spectral_components/{j:06d}/detector_ideal',scan_index=scan_index) for j in range(2))
        np.testing.assert_allclose(r.read('detector_ideal',scan_index=scan_index),expected)


def test_multicolor_stokes_keeps_color_and_polarization_incoherent(tmp_path):
    """Retain separate coherent carriers for mixed colors and polarization."""
    c=config_for(tmp_path/'color_stokes.h5')
    c.propagation.propagator_method='Stokes'
    c.propagation.propagator_config={'propagate':True,'input_stokes':(1.,0.,0.,0.)}
    c.illumination.spectral_components=(sim.SpectralComponentConfig(weight=.9),
        sim.SpectralComponentConfig(energy_factor=2.,weight=.1))
    c.outputs.save=('exit_wave','exit_stokes','fft_intensity','detector_ideal')
    r=sim.simulate_experiment(c)
    wave=r.read('exit_wave')
    assert wave.shape==(16,20,2,2,2)
    np.testing.assert_allclose(np.sum(abs(wave)**2,axis=(2,3,4)),r.read('exit_stokes')[...,0],rtol=1e-12)
