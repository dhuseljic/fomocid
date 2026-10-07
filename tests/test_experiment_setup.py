"""Shared experiment lifecycle, configuration isolation and optical checks."""
from pathlib import Path
import ast
import json
import numpy as np
import pytest
from scattering_calculator import simulation_pipelines as sim
from scattering_calculator.simulation_pipelines.examples import magnon_experiment


def small_config():
    """Store the standard experiment declaration.

    Parameters
    ----------
    None
        Uses the stored experiment state.

    Returns
    -------
    ExperimentConfig
        Small rectangular experiment used for independent optical checks.
    """
    c = magnon_experiment()
    return c.with_changes(
        simulation={"shape":(32,40),"real_space_pixel_size":4e-9},
        illumination={"illumination_config":{"center":(0.,0.),"distance":0.,"fwhm":70e-9,"alpha_beam":(0.,0.)}},
        propagation={"propagator_config":{"propagate":True}},
        detector={"shape":(12,14),"detector_center":(6,7),"analyzer_angle":None,"use_detector_pixel_footprint":False},
    )


def test_standard_setup_binds_grid_subdivides_and_preserves_incident_flux():
    """Assert standard setup binds grid subdivides and preserves incident flux.

    Parameters
    ----------
    None
        Uses the stored experiment state.

    Returns
    -------
    None
        Completes in place or raises if the asserted contract is violated.
    """
    c=small_config().with_changes(xray={"photon_flux":123.})
    run=sim.ScatteringExperiment(c).setup()
    assert run.sample.mask.shape==(15,32,40)
    np.testing.assert_allclose(sum(run.sample.layer_thicknesses),30e-9,rtol=1e-14)
    assert max(run.sample.layer_thicknesses)<=2e-9*(1+1e-12)
    np.testing.assert_allclose(np.sum(abs(run.illumination.illumination.illumination_jones)**2),123.)
    np.testing.assert_allclose(np.linalg.norm(run.sample.magnetization[0],axis=-1),1.,atol=1e-14)
    assert np.all(run.sample.mask==1)  # no aperture must retain the Co, not remove it
    run.propagate()
    assert run.detect().shape==(12,14)


def test_changes_are_independent_and_energy_reloads_material_and_q_grid():
    """Assert changes are independent and energy reloads material and q grid.

    Parameters
    ----------
    None
        Uses the stored experiment state.

    Returns
    -------
    None
        Completes in place or raises if the asserted contract is violated.
    """
    c=small_config()
    altered=c.with_changes(xray={"energy":790.},magnetic_pattern={"pattern_config":{**c.magnetic_pattern.pattern_config,'amplitude':.2}})
    assert c.xray.energy==778. and c.magnetic_pattern.pattern_config['amplitude']==.1
    a,b=sim.ScatteringExperiment(c).setup(),sim.ScatteringExperiment(altered).setup()
    assert not np.allclose(a.sample.layer_refractive_indices,b.sample.layer_refractive_indices,rtol=1e-6,atol=0)
    assert not np.array_equal(a.config.detector.detector_layout.detqx,b.config.detector.detector_layout.detqx)
    assert c.sample.sample_shape is None  # setup does not mutate the definition


@pytest.mark.parametrize('method',['fraunhofer','rayleigh_sommerfeld'])
@pytest.mark.parametrize('oversampling',[1,2])
def test_rotated_analyzer_completeness_and_unmodulated_extinction(method,oversampling):
    """Assert rotated analyzer completeness and unmodulated extinction.

    Parameters
    ----------
    method : str
        Detector propagation backend under test.
    oversampling : int
        Linear FFT padding factor used in the optical check.

    Returns
    -------
    None
        Completes in place or raises if the asserted contract is violated.
    """
    angle=.37
    c=small_config().with_changes(xray={"linear_polarization_angle":angle},
        propagation={"propagator_config":{"propagate":True,"farfield_oversampling":oversampling}},
        detector={"detector_propagation_method":method})
    run=sim.ScatteringExperiment(c).setup();exit_field=run.propagate().exit_wave
    def detect(analyzer):
        """Detect the configured analyzer image.

        Parameters
        ----------
        analyzer : float or None
            Analyzer angle in radians, or None for unfiltered detection.

        Returns
        -------
        ndarray
            Ideal or measured detector intensity on the configured camera grid.
        """
        cfg=c.with_changes(detector={"analyzer_angle":analyzer})
        return sim.ScatteringExperiment.from_exit_wave(cfg,exit_field).detect()
    total,parallel,crossed=detect(None),detect(angle),detect(angle+np.pi/2)
    np.testing.assert_allclose(parallel+crossed,total,rtol=2e-12,atol=1e-15)
    zero=c.with_changes(magnetic_pattern={"pattern_config":{**c.magnetic_pattern.pattern_config,'amplitude':0.}},
                        detector={"analyzer_angle":angle+np.pi/2})
    control=sim.ScatteringExperiment(zero).setup();control.propagate()
    assert control.detect().max()<total.max()*1e-20


def test_exit_only_run_can_continue_to_detector_and_mixed_stokes_preserves_power():
    """Assert exit only run can continue to detector and mixed stokes preserves power.

    Parameters
    ----------
    None
        Uses the stored experiment state.

    Returns
    -------
    None
        Completes in place or raises if the asserted contract is violated.
    """
    c=small_config().with_changes(propagation={"propagator_config":{
        "propagate":True,"calculate_farfield":False,"farfield_oversampling":2}})
    run=sim.ScatteringExperiment(c).setup();run.propagate()
    assert run.wavefront.hologram is None
    assert np.isfinite(run.detect()).all()
    assert run.wavefront.exit_wave_for_farfield.shape[:2]==(64,80)
    mixed=c.with_changes(xray={"linear_polarization_angle":None,"pol":"CR"},
        propagation={"propagator_method":"Stokes","propagator_config":{
            "propagate":True,"input_stokes":[1.,0.,0.,.6],"farfield_oversampling":2}})
    def detect(angle):
        """Detect the configured analyzer image.

        Parameters
        ----------
        angle : float
            Laboratory analyzer angle in radians from +x towards +y.

        Returns
        -------
        ndarray
            Ideal or measured detector intensity on the configured camera grid.
        """
        return sim.ScatteringExperiment(mixed.with_changes(detector={"analyzer_angle":angle})).run()
    # Separate Stokes propagation runs can differ at the 1e-12 relative level
    # from FFT/summation roundoff; conservation remains far tighter than 1e-10.
    np.testing.assert_allclose(detect(.23)+detect(.23+np.pi/2),detect(None),rtol=1e-11,atol=1e-15)


def test_common_config_matches_hdf5_backend_parameters_without_silent_loss():
    """Assert common config matches hdf5 backend parameters without silent loss.

    Parameters
    ----------
    None
        Uses the stored experiment state.

    Returns
    -------
    None
        Completes in place or raises if the asserted contract is violated.
    """
    c=small_config().with_changes(
        xray={"linear_polarization_angle":None,"pol":"CR"},sample={"max_slice_thickness":None},
        simulation={"other_config":{"grid_mode":"detector","oversampling":3,"random_seed":12}},
        detector={"analyzer_angle":None},
    )
    legacy=c.to_pipeline_config()
    assert legacy.recipe==c.sample.recipe
    assert legacy.xray_energy==c.xray.energy
    assert legacy.detector_shape==c.detector.shape
    assert legacy.oversampling==3 and legacy.random_seed==12
    with pytest.raises(ValueError,match='analyzers'):
        c.with_changes(detector={'analyzer_angle':.5}).to_pipeline_config()


def test_maintained_notebooks_do_not_reintroduce_example_specific_config_classes():
    """Assert maintained notebooks do not reintroduce example specific config classes.

    Parameters
    ----------
    None
        Uses the stored experiment state.

    Returns
    -------
    None
        Completes in place or raises if the asserted contract is violated.
    """
    root=Path(__file__).resolve().parents[1]
    forbidden={'FTHConfig','SkyrmionConfig','DetectorEffectsConfig','DetectorGeometry','HologramPipelineConfig','Structure'}
    paths=[* (root/'tutorials').glob('*.ipynb'),* (root/'paper/scattering_calculator/notebooks').glob('*.ipynb')]
    for path in paths:
        notebook=json.loads(path.read_text())
        code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        parse='\n'.join('pass' if line.lstrip().startswith(('%','!')) else line for line in code.splitlines())
        tree=ast.parse(parse)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=getattr(node.func,'id',getattr(node.func,'attr',''))
                assert name not in forbidden,(path.name,name)
                if name=='MagneticPatternConfig':
                    aliases={'stripe_width','lattice_spacing','skyr_radius','screening_radius',
                             'angle_stripes','wave_angle','oop_amplitude','diameter_spread'}
                    for keyword in node.keywords:
                        if keyword.arg=='pattern_config' and isinstance(keyword.value,ast.Dict):
                            keys={key.value for key in keyword.value.keys if isinstance(key,ast.Constant)}
                            assert not keys & aliases,(path.name,'use shared magnetic parameter names',keys & aliases)
            if isinstance(node,ast.ClassDef):
                assert node.name!='Config',(path.name,'notebook-local Config')
