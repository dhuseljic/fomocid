"""Timed-out simulations leave no partial data and do not stop the sweep."""
import multiprocessing as mp
import runpy
import time

import h5py
import numpy as np
import pytest

from scattering_calculator.simulation_pipelines.pipelines.hologram_pipeline import HologramPipeline, HologramPipelineRanges

pytestmark = pytest.mark.skipif('fork' not in mp.get_all_start_methods(), reason='Unix timeout isolation')


class SlowFirstPipeline(HologramPipeline):
    """Write temporary partial data before stalling the first experiment."""
    def _simulate_one(self, h5, idx):
        group = h5.create_group(f'{idx:05d}')
        group.create_group('metadata')
        group.create_dataset('payload', data=idx)
        if idx == 0:
            h5.flush()
            time.sleep(5)


def test_timeout_discards_partial_run_and_continues(tmp_path):
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    output = tmp_path/'timeout.h5'
    start = time.monotonic()
    SlowFirstPipeline(config=factory(), ranges=HologramPipelineRanges(), output_path=output,
        n_samples=2, verbose=False, experiment_timeout_seconds=.3).run()
    assert time.monotonic()-start < 4
    with h5py.File(output) as h5:
        assert '00000' not in h5
        assert h5['00001/payload'][()] == 1
        assert h5.attrs['acquisition_pair_count'] == 1
        assert not any('timed_out' in key or 'timeout' in key for key in h5.attrs)


def test_isolated_success_matches_ideal_and_sampler_advances_on_grid_retry(tmp_path):
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    config = factory(magnetic_pattern_max_grid_axis=512)
    direct, isolated = tmp_path/'direct.h5', tmp_path/'isolated.h5'
    HologramPipeline(config=config, ranges=HologramPipelineRanges(), output_path=direct,
        n_samples=1, verbose=False).run()
    draws = iter([.05, .005])
    HologramPipeline(config=config, ranges=HologramPipelineRanges(detector_distance=lambda _: next(draws)),
        output_path=isolated, n_samples=1, verbose=False, experiment_timeout_seconds=10).run()
    with h5py.File(direct) as a, h5py.File(isolated) as b:
        assert b['00000/metadata/sampling_attempts'][()] == 2
        for pol in ('CR', 'CL'):
            np.testing.assert_array_equal(a[f'00000/{pol}/ideal'][()], b[f'00000/{pol}/ideal'][()])


def test_isolated_tree_retains_leaf_numbering_and_parent_parameters(tmp_path):
    from scattering_calculator.simulation_pipelines.pipelines.hierarchical_hologram_pipeline import HierarchicalHologramPipeline
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    distances = iter([.005, .006])
    output = tmp_path/'tree.h5'
    HierarchicalHologramPipeline(config=factory(),
        ranges=HologramPipelineRanges(detector_distance=lambda _: next(distances)),
        output_path=output, n_samples=2, verbose=False, experiment_timeout_seconds=10,
        acquisition_variants=[{}, {'measurement_config': {'exposure_time': .2}}]).run()
    with h5py.File(output) as h5:
        assert [key for key in h5 if key.isdigit()] == [f'{i:05d}' for i in range(4)]
        for i in range(4):
            assert h5[f'{i:05d}/metadata/hierarchy/pattern_id'][()] == i//2
            assert h5[f'{i:05d}/metadata/detector/sample_to_detector_distance'][()] == [.005, .006][i//2]


class AbruptFirstPipeline(SlowFirstPipeline):
    """Simulate a native worker crash, with no exception sent through the pipe."""
    def _simulate_one(self, h5, idx):
        if idx == 0:
            import os
            os._exit(9)
        super()._simulate_one(h5, idx)


def test_abrupt_worker_exit_does_not_kill_sweep(tmp_path):
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    path = tmp_path/'abrupt.h5'
    AbruptFirstPipeline(config=factory(), ranges=HologramPipelineRanges(), output_path=path,
        n_samples=2, verbose=False, experiment_timeout_seconds=2).run()
    with h5py.File(path) as h5:
        assert '00000' not in h5 and '00001' in h5


def test_compatible_distance_is_saved_once_for_all_tree_leaves(tmp_path):
    from scattering_calculator.simulation_pipelines.pipelines.hierarchical_hologram_pipeline import HierarchicalHologramPipeline
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    output = tmp_path/'compatible-tree.h5'
    pipeline = HierarchicalHologramPipeline(
        config=factory(magnetic_pattern_distance_min=.005, magnetic_pattern_max_grid_axis=512),
        ranges=HologramPipelineRanges(detector_distance=lambda _: .4),
        output_path=output, n_samples=1, verbose=False, experiment_timeout_seconds=10,
        acquisition_variants=[{}, {'measurement_config': {'exposure_time': .2}}])
    pipeline.run()
    with h5py.File(output) as h5:
        distances = []
        for key in ('00000', '00001'):
            meta = h5[f'{key}/metadata/detector']
            distances.append(meta['sample_to_detector_distance'][()])
            assert meta['sampled_distance_m'][()] == .4
            assert meta['distance_adjusted_for_magnetic_grid'][()]
            assert meta['compatible_distance_min_m'][()] == .005
            assert h5[f'{key}/metadata/sampling_attempts'][()] == 1
        assert .005 <= distances[0] < .4
        assert distances[0] == distances[1]
