"""Common isotropic scalar prefixes are computed once for paired helicities."""
import runpy
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from scattering_calculator.beam_propagator.simple_propagation import scalar_wavefronts
from scattering_calculator.beam_propagator import simple_propagation
from scattering_calculator.simulation_pipelines.pipelines.hologram_pipeline import HologramPipeline, HologramPipelineRanges


class IndependentScalarPipeline(HologramPipeline):
    """Reference runner that always propagates each complete helicity stack."""
    def _setup_propagator(self, propagator_config, pol):
        propagator_config.propagator_config.pop('scalar_prefix_cache', None)
        return super()._setup_propagator(propagator_config, pol)


@pytest.mark.parametrize('propagate', [False, True])
def test_prefix_reuse_matches_independent_complex_exit_waves(tmp_path, propagate):
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    config = factory(recipe='[Au(8)Cr(2)]x7/SiN(20)/Co(6)Ta(2)Ir(3)', propagate=propagate)
    cached, reference = tmp_path/'cached.h5', tmp_path/'reference.h5'
    calls = []
    original = scalar_wavefronts.apply_compact_index_slice
    def tracked(self, field, stack, index, *args, **kwargs):
        calls.append(index)
        return original(self, field, stack, index, *args, **kwargs)
    with patch.object(scalar_wavefronts, 'apply_compact_index_slice', tracked), \
         patch.object(simple_propagation, 'label', wraps=simple_propagation.label) as geometry_labels:
        HologramPipeline(config, HologramPipelineRanges(), cached, 1, verbose=False).run()
    assert calls == list(range(9)) + [8]
    assert geometry_labels.call_count == 1
    IndependentScalarPipeline(config, HologramPipelineRanges(), reference, 1, verbose=False).run()
    with h5py.File(cached) as a, h5py.File(reference) as b:
        assert a['00000/metadata/propagator/CR/reused_prefix_layers'][()] == 0
        assert a['00000/metadata/propagator/CL/reused_prefix_layers'][()] == 8
        assert 'scalar_prefix_cache' not in a['00000/metadata/propagator_config']
        for pol in ('CR', 'CL'):
            times = a[f'00000/metadata/propagator/{pol}/timing_seconds']
            assert times['polarization_suffix'][()] > 0
            assert times['farfield'][()] > 0
        assert a['00000/metadata/propagator/CR/timing_seconds/shared_prefix'][()] > 0
        assert a['00000/metadata/propagator/CL/timing_seconds/shared_prefix'][()] == 0
        for pol in ('CR', 'CL'):
            for source in ('exit_wave', 'ideal'):
                np.testing.assert_array_equal(a[f'00000/{pol}/{source}'][()], b[f'00000/{pol}/{source}'][()])


@pytest.mark.parametrize('polarizations,layer', [(('CR', 'CL'), 1), (('LH', 'LV'), 2)])
def test_geometry_cache_keeps_polarization_dependent_indices_separate(polarizations, layer):
    from types import SimpleNamespace
    shape = (3, 8, 10)
    magnetization = np.zeros((*shape, 3))
    magnetization[..., 2] = 1.
    magnetization[..., 0] = 1.
    indices = np.array([[.99+.001j, 0j, 0j],
                        [.98+.002j, .001+.0002j, 0j],
                        [.97+.003j, 0j, .002+.0004j]])
    sample = SimpleNamespace(mask=np.ones(shape), magnetization=magnetization,
                             layer_refractive_indices=indices)
    cache = {}
    first = simple_propagation.calculate_scalar_refractive_index_stack(
        sample, polarizations[0], geometry_cache=cache)
    before = first.materialize_layer(layer).copy()
    second = simple_propagation.calculate_scalar_refractive_index_stack(
        sample, polarizations[1], geometry_cache=cache)
    assert first is not second
    assert first.aperture_support is second.aperture_support
    np.testing.assert_array_equal(first.materialize_layer(0), second.materialize_layer(0))
    assert not np.array_equal(first.materialize_layer(layer), second.materialize_layer(layer))
    np.testing.assert_array_equal(first.materialize_layer(layer), before)
    for stack, pol in [(first, polarizations[0]), (second, polarizations[1])]:
        circular, linear = simple_propagation.scalar_refractive_index_coefficients(pol)
        expected = indices[layer, 0] + circular*indices[layer, 1] + linear*indices[layer, 2]
        np.testing.assert_allclose(stack.materialize_layer(layer), expected)
