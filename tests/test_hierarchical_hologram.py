"""Hierarchical branching reuses physics and retains canonical output."""
import runpy
from unittest.mock import patch

import pytest
import h5py
import numpy as np

from scattering_calculator.simulation_pipelines.pipelines.hierarchical_hologram_pipeline import HierarchicalHologramPipeline
from scattering_calculator.simulation_pipelines.pipelines.hologram_pipeline import HologramPipelineRanges
from scattering_calculator.simulation_pipelines.simulation_configuration import MagneticPatternConfig, SamplePropagatorConfig, DetectorConfig


def test_tree_reuses_pattern_exit_waves_and_ideal_projection(tmp_path):
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    path = tmp_path / 'tree.h5'
    counts = dict(pattern=0, exit=0, ideal=0)
    def track(method, key):
        def wrapped(self, *args, **kwargs):
            counts[key] += 1
            return method(self, *args, **kwargs)
        return wrapped
    with patch.object(MagneticPatternConfig, 'create_pattern', track(MagneticPatternConfig.create_pattern, 'pattern')), \
         patch.object(SamplePropagatorConfig, 'setup', track(SamplePropagatorConfig.setup, 'exit')), \
         patch.object(DetectorConfig, 'detect_hologram', track(DetectorConfig.detect_hologram, 'ideal')):
        HierarchicalHologramPipeline(config=factory(), ranges=HologramPipelineRanges(),
            output_path=path, n_samples=1, verbose=False,
            exit_variants=[{}, {'xray_energy': 780.}],
            geometry_variants=[{}, {'sample_to_detector_distance': .006}],
            acquisition_variants=[{'measurement_config': {'exposure_time': .1}},
                                  {'measurement_config': {'exposure_time': 1.}}]).run()
    assert counts == dict(pattern=1, exit=4, ideal=8)
    with h5py.File(path) as h5:
        assert len([k for k in h5 if k.isdigit()]) == 8
        assert h5['00007/metadata/hierarchy/pattern_id'][()] == 0
        assert h5['00007/metadata/hierarchy/reused_ideal_hologram'][()]
        assert h5['00000/metadata/sample/real_space_pixel_size'][()] == h5['00004/metadata/sample/real_space_pixel_size'][()]
        for pol in ['CR', 'CL']:
            np.testing.assert_array_equal(h5[f'00000/{pol}/ideal'][()], h5[f'00001/{pol}/ideal'][()])
            np.testing.assert_array_equal(h5[f'00000/{pol}/exit_wave'][()], h5[f'00002/{pol}/exit_wave'][()])
            assert not np.array_equal(h5[f'00000/{pol}/ideal'][()], h5[f'00002/{pol}/ideal'][()])
            assert not np.array_equal(h5[f'00000/{pol}/detected'][()], h5[f'00001/{pol}/detected'][()])
        assert h5['00000/metadata/measurement_config/exposure_time'][()] == .1
        assert h5['00001/metadata/measurement_config/exposure_time'][()] == 1.


def test_first_branch_matches_independent_pipeline_and_rejects_oh_changes(tmp_path):
    import pytest
    from scattering_calculator.simulation_pipelines.pipelines.hologram_pipeline import HologramPipeline
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    common = dict(config=factory(), ranges=HologramPipelineRanges(), n_samples=1, verbose=False)
    direct, tree = tmp_path / 'direct.h5', tmp_path / 'tree.h5'
    HologramPipeline(output_path=direct, **common).run()
    HierarchicalHologramPipeline(output_path=tree, **common).run()
    with h5py.File(direct) as a, h5py.File(tree) as b:
        for pol in ['CR', 'CL']:
            for source in ['exit_wave', 'ideal']:
                np.testing.assert_array_equal(a[f'00000/{pol}/{source}'][()], b[f'00000/{pol}/{source}'][()])
    with pytest.raises(ValueError, match='object-hole'):
        HierarchicalHologramPipeline(output_path=tmp_path/'bad.h5',
            exit_variants=[{'aperture_radii': [40e-9, 6e-9]}], **common).run()


def test_distance_is_sampled_per_parent_and_shared_by_its_leaves(tmp_path):
    from scattering_calculator.simulation_pipelines import Uniform
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    path = tmp_path / 'parent_distances.h5'
    HierarchicalHologramPipeline(config=factory(),
        ranges=HologramPipelineRanges(detector_distance=Uniform(.005, .006)),
        output_path=path, n_samples=2, verbose=False,
        exit_variants=[{}, {'xray_energy': 780.}],
        acquisition_variants=[{}, {'measurement_config': {'exposure_time': .2}}]).run()
    with h5py.File(path) as h5:
        distances, pitches = {}, {}
        for key in h5:
            if not key.isdigit():
                continue
            meta = h5[f'{key}/metadata']
            parent = int(meta['hierarchy/pattern_id'][()])
            distances.setdefault(parent, set()).add(float(meta['detector/sample_to_detector_distance'][()]))
            pitches.setdefault(parent, set()).add(float(meta['sample/real_space_pixel_size'][()]))
        assert len(distances) == 2
        assert all(len(values) == 1 for values in distances.values())
        assert all(len(values) == 1 for values in pitches.values())
        assert distances[0] != distances[1]


@pytest.mark.parametrize('hierarchical', [False, True])
def test_oversized_draw_is_resampled_before_pattern_generation(tmp_path, hierarchical):
    from scattering_calculator.simulation_pipelines.pipelines.hologram_pipeline import HologramPipeline
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    distances = iter([.05, .005])
    pipeline_type = HierarchicalHologramPipeline if hierarchical else HologramPipeline
    path = tmp_path / 'resampled.h5'
    with patch.object(MagneticPatternConfig, 'create_pattern', autospec=True,
                      side_effect=MagneticPatternConfig.create_pattern) as generate:
        pipeline_type(config=factory(magnetic_pattern_max_grid_axis=512),
            ranges=HologramPipelineRanges(detector_distance=lambda _: next(distances)),
            output_path=path, n_samples=1, verbose=False, max_sampling_attempts=2).run()
        assert generate.call_count == 1
    with h5py.File(path) as h5:
        assert h5.attrs['sampling_rejected_draws'] == 1
        assert h5.attrs['sampling_completed_parents'] == 1
        assert h5['00000/metadata/sampling_attempts'][()] == 2
        assert h5['00000/metadata/detector/sample_to_detector_distance'][()] == .005


@pytest.mark.parametrize('hierarchical', [False, True])
def test_impossible_grid_is_skipped_after_bounded_attempts(tmp_path, hierarchical):
    from scattering_calculator.simulation_pipelines.pipelines.hologram_pipeline import HologramPipeline
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    path = tmp_path / 'skipped.h5'
    pipeline_type = HierarchicalHologramPipeline if hierarchical else HologramPipeline
    pipeline_type(config=factory(magnetic_pattern_max_grid_axis=1),
        ranges=HologramPipelineRanges(), output_path=path, n_samples=2,
        verbose=False, max_sampling_attempts=2).run()
    with h5py.File(path) as h5:
        assert h5.attrs['sampling_skipped_parents'] == 2
        assert h5.attrs['sampling_rejected_draws'] == 4
        assert h5.attrs['acquisition_pair_count'] == 0
