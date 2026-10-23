"""Headless checks for desktop configuration, background work and result loading."""
import multiprocessing as mp
import runpy
from dataclasses import fields

import h5py
import numpy as np
import pytest

from scattering_calculator.gui.single_experiment import (
    default_config, group_for, parse_value, read_result, simulate_to_file, SECTIONS,
    SaturatingResponse, worker,
)


def test_all_configuration_fields_have_a_settings_group_and_parseable_default():
    config = default_config()
    for field in fields(config):
        assert group_for(field.name) in SECTIONS
        value = getattr(config, field.name)
        text = value if isinstance(value, str) else repr(value)
        assert parse_value(text, value) == value
    assert parse_value('None', None) is None
    assert parse_value('(64, 80)', (256, 256)) == (64, 80)
    for text, template in [('2.5', 1), ('[3, 4]', (3, 4)), ('1', True), ('True', 1.),
                           ('__import__("os").system("echo unsafe")', {})]:
        with pytest.raises(ValueError):
            parse_value(text, template)


def test_background_process_produces_the_four_views_and_preserves_metadata(tmp_path):
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    config = factory()
    config.detector_params['nonlinear_response'] = SaturatingResponse(50000.)
    output, log = tmp_path/'single.h5', tmp_path/'single.log'
    context = mp.get_context('spawn')
    messages = context.Queue()
    process = context.Process(target=worker, args=(config, output, log, messages))
    process.start()
    try:
        kind, payload = messages.get(timeout=45)
        process.join(timeout=5)
        assert process.exitcode == 0
        assert kind == 'done', payload
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        messages.close()
    result = read_result(output)
    assert result['magnetic'].ndim == 2
    assert result['recipe'] == config.recipe
    for pol in ('CR', 'CL'):
        assert np.iscomplexobj(result[pol]['exit_wave'])
        assert result[pol]['ideal'].shape == (32, 32)
        assert result[pol]['detected'].shape == (32, 32)
    with h5py.File(output) as h5:
        assert h5['00000/metadata/detector_response/model'].asstr()[()] == 'nonlinear'
        assert h5['00000/metadata/gui/response_scale_adu'][()] == 50000.
        meta = h5['00000/metadata/sample/magnetic_pattern']
        assert 'binary_pattern' not in meta and 'analysis_mask' not in meta
    assert 'timing:' in log.read_text()


def test_single_experiment_grid_failure_is_reported_without_resampling(tmp_path):
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    with pytest.raises(ValueError, match='allocation limit'):
        simulate_to_file(factory(magnetic_pattern_max_grid_axis=1), tmp_path/'too_large.h5')


def test_all_four_plots_render_without_a_display(tmp_path):
    from types import SimpleNamespace
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from scattering_calculator.gui.single_experiment import SimulatorApp
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    result = {'magnetic': np.ones((8, 12))}
    for pol, multiplier in [('CR', 2.), ('CL', 3.)]:
        result[pol] = dict(exit_wave=np.ones((16, 20), complex)*multiplier*(1+1j),
                          ideal=np.arange(80).reshape(8, 10)*multiplier,
                          detected=np.ones((8, 10))*multiplier)
    view = SimpleNamespace(figure=figure, canvas=canvas, result=result,
        polarization=SimpleNamespace(get=lambda: 'CR−CL'),
        exit_view=SimpleNamespace(get=lambda: 'phase'),
        log_view=SimpleNamespace(get=lambda: True))
    SimulatorApp.draw(view)
    canvas.draw()
    assert len(figure.axes) == 8  # Four images and four colorbars.


def test_geometry_acquisition_and_sample_grid_are_separate():
    assert group_for('pattern_config') == 'Magnetic pattern'
    assert group_for('oversampling') == 'Sample structure and grid'
    assert group_for('recipe') == 'Sample structure and grid'
    assert group_for('detector_pixel_size') == 'Detector geometry'
    assert group_for('detector_distance') == 'Detector geometry'
    assert group_for('measurement_config') == 'Acquisition and sensor response'
    assert group_for('detector_params') == 'Acquisition and sensor response'
    assert group_for('artifacts_config') == 'Acquisition and sensor response'
    assert group_for('illumination_fwhm') == 'X-rays and illumination'
    assert group_for('beamstop_config') == 'Beamstop'
    assert group_for('propagate') == 'Propagation'


def test_aperture_preview_is_a_depth_weighted_open_fraction():
    from scattering_calculator.gui.single_experiment import aperture_preview
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    config = factory(aperture_top_radius_factors=[1., 1.])
    image, extent = aperture_preview(config, max_axis=128)
    assert image.shape == (128, 128)
    assert np.all((image >= 0) & (image <= 1))
    # OH opens only the 10 nm Au mask; RH opens the complete 36 nm stack.
    np.testing.assert_allclose(image[64, 64], 10/36, atol=1e-5)
    pitch_um = (extent[1]-extent[0])/image.shape[1]
    rh_y = int(round(config.aperture_centers[1][0]*1e6/pitch_um+64))
    rh_x = int(round(config.aperture_centers[1][1]*1e6/pitch_um+64))
    assert image[rh_y, rh_x] > .95
    assert image[0, 0] == 0


def test_custom_plot_colors_preserve_zoom_and_wheel_zoom_changes_limits():
    from types import SimpleNamespace
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from scattering_calculator.gui.single_experiment import SimulatorApp
    figure = Figure()
    result = {'magnetic': np.ones((8, 12))}
    for pol in ('CR', 'CL'):
        result[pol] = dict(exit_wave=np.ones((8, 12), complex),
                          ideal=np.arange(96).reshape(8, 12),
                          detected=np.ones((8, 12)))
    view = SimpleNamespace(figure=figure, canvas=FigureCanvasAgg(figure), result=result,
        polarization=SimpleNamespace(get=lambda: 'CR'),
        exit_view=SimpleNamespace(get=lambda: 'amplitude'),
        log_view=SimpleNamespace(get=lambda: True),
        plot_settings={2: dict(cmap='viridis', scale='linear', vmin=10., vmax=80.)})
    SimulatorApp.draw(view)
    image = view.image_axes[2].images[0]
    assert image.get_cmap().name == 'viridis'
    assert image.get_clim() == (10., 80.)
    view.image_axes[2].set_xlim(2, 6)
    SimulatorApp.draw(view)
    assert view.image_axes[2].get_xlim() == (2., 6.)
    SimulatorApp.scroll_zoom(view, SimpleNamespace(inaxes=view.image_axes[2], xdata=4., ydata=4., button='up'))
    left, right = view.image_axes[2].get_xlim()
    assert right-left < 4


def test_irrelevant_legacy_and_preparation_controls_are_hidden():
    from scattering_calculator.gui.single_experiment import parameter_is_active
    assert not parameter_is_active(('pattern_config_length',))
    assert not parameter_is_active(('detector_noise_rms',))
    assert not parameter_is_active(('pattern_config', 'initial_order'))
    assert not parameter_is_active(('pattern_config', 'worm_waviness'))
    assert not parameter_is_active(('pattern_config', 'initial_domain_count'), preparation='hexagonal', order=1.)
    assert parameter_is_active(('pattern_config', 'initial_order'), preparation='hexagonal')
    assert parameter_is_active(('pattern_config', 'worm_waviness'), preparation='worms', order=.5)
    assert not parameter_is_active(('pattern_config', 'worm_length'), preparation='worms', order=0.)
    assert not parameter_is_active(('pattern_config', 'disorder_length'), disorder=0.)
    assert parameter_is_active(('pattern_config', 'disorder_length'), disorder=.2)
    assert not parameter_is_active(('pattern_config', 'length_tolerance'), attempts=1)
    assert parameter_is_active(('pattern_config', 'length_tolerance'), attempts=3)


def test_polarization_ratio_preserves_complex_phase_and_masks_undefined_values():
    from scattering_calculator.gui.single_experiment import polarization_view
    cr = np.array([1+1j, 0j, np.nan+0j, 2+0j])
    cl = np.array([2j, 1j, 1j, np.inf+0j])
    ratio = polarization_view(cr, cl, 'CL/CR')
    np.testing.assert_allclose(ratio[0], 1+1j)
    assert np.isnan(ratio[1:]).all()
    np.testing.assert_array_equal(polarization_view(cr[:1], cl[:1], 'CL−CR'), cl[:1]-cr[:1])
    np.testing.assert_array_equal(polarization_view(cr[:1], cl[:1], 'CR'), cr[:1])
    np.testing.assert_array_equal(polarization_view(cr[:1], cl[:1], 'CL'), cl[:1])
    real = polarization_view(np.array([2, 0]), np.array([4, 1]), 'CL/CR')
    assert real[0] == 2 and np.isnan(real[1])


def test_ratio_phase_keeps_complex_contrast_and_autoscales_small_phase():
    from types import SimpleNamespace
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from scattering_calculator.gui.single_experiment import SimulatorApp
    phase = np.linspace(-.003, .003, 96).reshape(8, 12)
    cr = np.ones_like(phase, dtype=complex)*(1+2j)
    cl = cr * 1.01 * np.exp(1j*phase)
    result = dict(magnetic=np.ones_like(phase))
    for pol, wave in [('CR', cr), ('CL', cl)]:
        result[pol] = dict(exit_wave=wave, ideal=np.ones_like(phase), detected=np.ones_like(phase))
    figure = Figure()
    view = SimpleNamespace(figure=figure, canvas=FigureCanvasAgg(figure), result=result,
        polarization=SimpleNamespace(get=lambda: 'CL/CR'),
        exit_view=SimpleNamespace(get=lambda: 'phase'), log_view=SimpleNamespace(get=lambda: False))
    SimulatorApp.draw(view)
    image = view.image_axes[1].images[0]
    np.testing.assert_allclose(image.get_array(), phase, atol=1e-14)
    np.testing.assert_allclose(image.get_clim(), [-.003, .003], atol=1e-14)
    view.exit_view = SimpleNamespace(get=lambda: 'amplitude')
    SimulatorApp.draw(view)
    np.testing.assert_allclose(view.image_axes[1].images[0].get_array(), 1.01)
