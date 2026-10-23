"""Known geometries exercise counts, orientational order and HDF5 persistence."""
import h5py
import numpy as np
import pytest

from scattering_calculator.sample_generator.domain_analysis import (
    classify_magnetic_domains, write_domain_classification,
)


def test_saturated_and_unresolved_are_distinguishable():
    pattern = np.ones((64, 64))
    result = classify_magnetic_domains(pattern)
    assert result['state'] == 'saturated'
    assert result['bubble_count'] == 0
    assert result['stripe_count'] == 1
    assert result['stripe_nematic_order'] is None
    pattern[20, 20] = -1
    result = classify_magnetic_domains(pattern)
    assert result['state'] == 'saturated'
    assert result['noise_count'] == 1
    assert result['minority_stripe_count'] == 0
    assert result['majority_stripe_count'] == 1


def test_counts_mixed_state_and_is_invariant_to_polarity():
    yy, xx = np.indices((128, 128))
    pattern = np.ones((128, 128))
    pattern[(yy-32)**2 + (xx-32)**2 <= 8**2] = -1
    pattern[70:78, 20:100] = -1
    for field in (pattern, -pattern):
        result = classify_magnetic_domains(field)
        assert result['state'] == 'mixed'
        assert result['bubble_count'] == 1
        assert result['stripe_count'] == 2


def test_parallel_and_crossed_stripe_axes():
    pattern = np.ones((128, 128))
    pattern[10:16, 10:55] = -1
    pattern[40:46, 10:55] = -1
    result = classify_magnetic_domains(pattern)
    assert result['stripe_count'] == 3
    parallel_order = result['stripe_nematic_order']
    assert parallel_order > .7
    pattern[40:46, 10:55] = 1
    pattern[60:105, 90:96] = -1
    assert classify_magnetic_domains(pattern)['stripe_nematic_order'] < parallel_order


def test_hexagonal_bubbles_and_random_centers():
    yy, xx = np.indices((240, 240))
    centers = [(y, x + (10 if row % 2 else 0))
               for row, y in enumerate(np.arange(24, 216, 10*np.sqrt(3)))
               for x in np.arange(24, 206, 20)]
    def field(points):
        pattern = np.ones((240, 240))
        for y, x in points:
            pattern[(yy-y)**2 + (xx-x)**2 <= 6**2] = -1
        return pattern
    ordered = classify_magnetic_domains(field(centers))
    jittered = np.asarray(centers) + np.random.default_rng(4).uniform(-3, 3, (len(centers), 2))
    disordered = classify_magnetic_domains(field(jittered))
    assert ordered['state'] == 'bubble'
    assert ordered['majority_stripe_count'] == 1
    assert ordered['bubble_count'] == len(centers)
    assert ordered['local_sixfold_order'] > .7
    assert ordered['sixfold_order'] > disordered['sixfold_order']
    # The connected majority background is classified and included too.
    assert ordered['stripe_count'] == 1


def test_hdf5_roundtrip_records_state_options_and_undefined_order(tmp_path):
    path = tmp_path / 'states.h5'
    pattern = np.ones((64, 64))
    pattern[20:26, 10:54] = -1
    with h5py.File(path, 'w') as h5:
        write_domain_classification(h5.create_group('classification'), pattern, min_area=12)
        write_domain_classification(h5.create_group('saturated'), np.ones((16, 16)))
    with h5py.File(path) as h5:
        group = h5['classification']
        assert group['state'].asstr()[()] == 'stripe'
        assert group['stripe_count'][()] == 2
        assert group['negative_stripe_count'][()] == 1
        assert group['positive_stripe_count'][()] == 1
        assert group['classification_min_area_px'][()] == 12
        assert 0 <= group['stripe_nematic_order'][()] <= 1
        assert np.isnan(group['sixfold_order'][()])
        assert np.isnan(h5['saturated/stripe_nematic_order'][()])


@pytest.mark.parametrize('pattern', [np.empty((0, 3)), np.full((3, 3), np.nan)])
def test_invalid_patterns_fail(pattern):
    with pytest.raises(ValueError):
        classify_magnetic_domains(pattern)


def test_oh_clipped_bubble_uses_full_shape_even_for_one_visible_pixel():
    yy, xx = np.indices((96, 96))
    pattern = np.ones((96, 96))
    pattern[(yy-48)**2 + (xx-48)**2 <= 10**2] = -1
    oh = xx < 39  # Only the leftmost pixel of the bubble is visible.
    result = classify_magnetic_domains(pattern, analysis_mask=oh)
    assert result['bubble_count'] == 1
    assert result['stripe_count'] == 1
    assert result['noise_count'] == 0
    component = result['minority_components'][0]
    assert component['visible_area'] == 1
    assert component['area'] > 9
    assert component['touches_aperture']
    assert not component['touches_border']
    # Full component labels are retained to measure order from full geometry.
    assert np.count_nonzero(result['minority_label_image']) == component['area']


def test_oh_selection_does_not_swap_domain_and_background():
    yy, xx = np.indices((96, 96))
    pattern = np.ones((96, 96))
    pattern[(yy-48)**2 + (xx-48)**2 <= 10**2] = -1
    oh = (yy-48)**2 + (xx-48)**2 <= 5**2
    result = classify_magnetic_domains(pattern, analysis_mask=oh)
    assert result['minority_polarity'] == -1
    assert result['bubble_count'] == 1
    assert result['state'] == 'bubble'


def test_stripe_connected_outside_oh_is_counted_once():
    pattern = np.ones((96, 96))
    pattern[15:80, 20:25] = -1
    pattern[15:80, 55:60] = -1
    pattern[15:20, 20:60] = -1  # Connection is outside the OH.
    oh = np.zeros_like(pattern, dtype=bool)
    oh[40:70, 10:70] = True
    result = classify_magnetic_domains(pattern, analysis_mask=oh)
    assert result['stripe_count'] == 2
    assert result['minority_stripe_count'] == 1
    assert result['bubble_count'] == 0


def test_domains_entirely_outside_oh_are_not_counted(tmp_path):
    yy, xx = np.indices((96, 96))
    pattern = np.ones((96, 96))
    pattern[(yy-70)**2 + (xx-70)**2 <= 10**2] = -1
    oh = xx < 30
    result = classify_magnetic_domains(pattern, analysis_mask=oh)
    assert result['state'] == 'saturated'
    assert result['bubble_count'] == 0
    assert result['stripe_count'] == result['majority_stripe_count'] == 1
    assert not result['minority_components']
    assert not np.any(result['minority_label_image'])
    with h5py.File(tmp_path / 'masked.h5', 'w') as h5:
        write_domain_classification(h5.create_group('classification'), pattern, analysis_mask=oh)
        assert h5['classification/bubble_count'][()] == 0
        assert h5['classification'].attrs['method'] == 'both_polarities_connected_components_v6'


def test_elliptical_bubble_and_elongated_stripe():
    yy, xx = np.indices((128, 128))
    pattern = np.ones((128, 128))
    pattern[((yy-30)/8)**2 + ((xx-40)/20)**2 <= 1] = -1
    result = classify_magnetic_domains(pattern)
    assert result['state'] == 'bubble'
    assert result['minority_bubble_count'] == 1
    assert result['minority_components'][0]['eccentricity'] > .8
    pattern[80:86, 10:110] = -1
    result = classify_magnetic_domains(pattern)
    assert result['bubble_count'] == 1
    assert result['stripe_count'] == 2


@pytest.mark.parametrize('shift', [(0, 0), (0, 64), (64, 0), (64, 64)])
def test_periodic_bubble_is_not_split_or_misclassified_at_edges(shift):
    yy, xx = np.indices((128, 128))
    pattern = np.ones((128, 128))
    pattern[((yy-64)/8)**2 + ((xx-64)/20)**2 <= 1] = -1
    pattern = np.roll(pattern, shift, axis=(0, 1))
    result = classify_magnetic_domains(pattern, periodic=True)
    assert result['bubble_count'] == 1
    assert result['stripe_count'] == result['majority_stripe_count'] == 1
    assert result['noise_count'] == 0
    assert result['minority_components'][0]['area'] == np.count_nonzero(pattern < 0)


def test_periodic_winding_stripes_are_not_bubbles():
    pattern = np.ones((96, 96))
    pattern[:, 10:18] = -1
    pattern[:, 50:58] = -1
    result = classify_magnetic_domains(pattern, periodic=True)
    assert result['stripe_count'] == 4
    assert result['positive_stripe_count'] == result['negative_stripe_count'] == 2
    assert result['bubble_count'] == 0


@pytest.mark.parametrize('magnetization', [-.6, .6])
def test_notebook_bubbles_include_majority_background(magnetization):
    from scattering_calculator.sample_generator import create_constrained_domain_pattern
    pattern, metadata = create_constrained_domain_pattern(
        (128, 128), target_magnetization=magnetization,
        characteristic_length=16., alignment=0., direction=0.,
        disorder=0., disorder_length=1., initial_state='random', initial_order=0.,
        wall_stiffness=2., n_steps=300, time_step=.2,
        length_tolerance=.15, max_length_iterations=1, seed=7)
    result = classify_magnetic_domains(pattern, periodic=True)
    assert result['state'] == 'bubble'
    assert result['stripe_count'] == result['majority_stripe_count'] == 1
    assert result['minority_stripe_count'] == 0
    assert result['bubble_count'] == result['minority_bubble_count'] == 46
    open_result = classify_magnetic_domains(pattern, periodic=False, boundary_context_periodic=True)
    assert metadata['measurements']['bubble_count'] == open_result['bubble_count']
    assert metadata['measurements']['stripe_count'] == open_result['stripe_count']


def test_periodic_hdf5_roundtrip(tmp_path):
    yy, xx = np.indices((64, 64))
    pattern = np.ones((64, 64))
    pattern[(yy-32)**2 + (xx-32)**2 <= 8**2] = -1
    pattern = np.roll(pattern, (32, 32), axis=(0, 1))
    with h5py.File(tmp_path / 'periodic.h5', 'w') as h5:
        write_domain_classification(h5.create_group('state'), pattern, periodic=True)
    with h5py.File(tmp_path / 'periodic.h5') as h5:
        assert h5['state/periodic'][()]
        assert h5['state/bubble_count'][()] == 1
        assert h5['state/stripe_count'][()] == 1
        assert 'periodic' in h5['state'].attrs['boundary_convention']


def test_majority_bubble_is_counted_and_totals_sum_both_polarities(tmp_path):
    yy, xx = np.indices((128, 128))
    radius2 = (yy-64)**2 + (xx-64)**2
    pattern = np.ones((128, 128))
    pattern[(radius2 <= 30**2) & (radius2 > 20**2)] = -1
    # Positive majority contains an interior bubble as well as the background.
    for field in (pattern, -pattern):
        result = classify_magnetic_domains(field)
        assert result['majority_bubble_count'] == 1
        assert result['majority_stripe_count'] == 1
        assert result['minority_bubble_count'] == 0
        assert result['minority_stripe_count'] == 1
        assert result['bubble_count'] == 1
        assert result['stripe_count'] == 2
        for kind in ('bubble', 'stripe', 'noise'):
            assert result[f'{kind}_count'] == (
                result[f'positive_{kind}_count'] + result[f'negative_{kind}_count'])
            assert result[f'{kind}_count'] == (
                result[f'majority_{kind}_count'] + result[f'minority_{kind}_count'])
    with h5py.File(tmp_path / 'both.h5', 'w') as h5:
        write_domain_classification(h5.create_group('state'), pattern)
    with h5py.File(tmp_path / 'both.h5') as h5:
        for kind in ('bubble', 'stripe', 'noise'):
            group = h5['state']
            assert group[f'{kind}_count'][()] == (
                group[f'positive_{kind}_count'][()] + group[f'negative_{kind}_count'][()])
    # Exclude the external positive background; retain the majority bubble.
    result = classify_magnetic_domains(pattern, analysis_mask=radius2 <= 29**2)
    assert result['majority_bubble_count'] == 1
    assert result['majority_stripe_count'] == 0
    assert result['bubble_count'] == result['stripe_count'] == 1


@pytest.mark.parametrize('polarity', [-1, 1])
@pytest.mark.parametrize('periodic', [False, True])
def test_homogeneous_pattern_is_one_saturated_domain(polarity, periodic):
    result = classify_magnetic_domains(np.full((16, 16), polarity), periodic=periodic)
    assert result['bubble_count'] == 0
    assert result['stripe_count'] == 1
    assert result['majority_stripe_count'] == 1
    assert result['minority_stripe_count'] == 0
    assert result['state'] == 'saturated'
    assert result['is_uniform']


def test_majority_background_sets_bubble_state_and_added_stripe_sets_mixed():
    yy, xx = np.indices((128, 128))
    pattern = np.ones((128, 128))
    pattern[(yy-32)**2 + (xx-32)**2 <= 8**2] = -1
    result = classify_magnetic_domains(pattern)
    assert (result['bubble_count'], result['stripe_count'], result['state']) == (1, 1, 'bubble')
    pattern[80:86, 20:100] = -1
    result = classify_magnetic_domains(pattern)
    assert (result['bubble_count'], result['stripe_count'], result['state']) == (1, 2, 'mixed')
    pattern[(yy-32)**2 + (xx-32)**2 <= 8**2] = 1
    result = classify_magnetic_domains(pattern)
    assert (result['bubble_count'], result['stripe_count'], result['state']) == (0, 2, 'stripe')


@pytest.mark.parametrize('parameter,values,expected', [
    ('max_length_iterations', [1, 2, 3, 5], [1, 5, 5, 5]),
    ('length_tolerance', [.05, .1, .15, .25], [5, 5, 5, 1]),
])
def test_open_notebook_calibration_bubble_counts(parameter, values, expected):
    from scattering_calculator.sample_generator import create_constrained_domain_pattern
    counts = []
    for value in values:
        controls = dict(target_magnetization=0., characteristic_length=16.,
                        n_steps=300, max_length_iterations=5, seed=7)
        controls[parameter] = value
        pattern, metadata = create_constrained_domain_pattern((128, 128), **controls)
        result = classify_magnetic_domains(pattern, periodic=False, boundary_context_periodic=True)
        counts.append(result['bubble_count'])
        assert metadata['measurements']['bubble_count'] == result['bubble_count']
    assert counts == expected


@pytest.mark.parametrize('order', [0., .25, .5, .75, 1.])
def test_hexagonal_initial_order_gallery_is_bubbles_on_one_background(order):
    from scattering_calculator.sample_generator import create_constrained_domain_pattern
    pattern, metadata = create_constrained_domain_pattern((128, 128),
        characteristic_length=16., target_magnetization=-.6,
        initial_state='hexagonal', initial_order=order,
        n_steps=300, max_length_iterations=1, seed=7)
    result = classify_magnetic_domains(pattern, periodic=False, boundary_context_periodic=True)
    assert result['state'] == 'bubble'
    assert result['bubble_count'] > 0
    assert result['stripe_count'] == result['majority_stripe_count'] == 1
    assert result['minority_stripe_count'] == 0
    assert result['boundary_fragment_count'] > 0
    assert metadata['measurements']['state'] == 'bubble'
    assert result['bubble_count'] == classify_magnetic_domains(pattern, periodic=False)['bubble_count']


def test_periodic_context_does_not_hide_real_winding_stripes():
    pattern = np.ones((96, 96))
    pattern[:, 10:18] = -1
    pattern[:, 50:58] = -1
    result = classify_magnetic_domains(pattern, boundary_context_periodic=True)
    assert result['state'] == 'stripe'
    assert result['stripe_count'] > 1
    assert result['boundary_fragment_count'] == 0


@pytest.mark.parametrize('visible_sides', ['left', 'right', 'both'])
def test_periodic_bubble_counts_once_only_when_visible_in_oh(visible_sides):
    yy, xx = np.indices((128, 128))
    pattern = np.ones((128, 128))
    # One bubble wraps across x=0; a second complete bubble is outside the OH.
    dx = np.minimum(xx, 128-xx)
    pattern[(yy-64)**2 + dx**2 <= 10**2] = -1
    pattern[(yy-30)**2 + (xx-64)**2 <= 8**2] = -1
    oh = np.zeros_like(pattern, dtype=bool)
    if visible_sides in ('left', 'both'):
        oh[58:71, :7] = True
    if visible_sides in ('right', 'both'):
        oh[58:71, -7:] = True
    result = classify_magnetic_domains(pattern, periodic=True, analysis_mask=oh)
    assert result['bubble_count'] == 1
    assert result['stripe_count'] == 0  # OH lies entirely inside that bubble.
    assert result['boundary_fragment_count'] == 0
    assert len(result['components']) == 1
    full_bubble = classify_magnetic_domains(pattern, periodic=True)
    assert full_bubble['bubble_count'] == 2
    assert result['components'][0]['area'] > result['components'][0]['visible_area']
    assert result['components'][0]['touches_border']
    assert result['components'][0]['is_bubble']
