"""Common isotropic scalar prefixes are computed once for paired helicities."""
import runpy
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from scattering_calculator.beam_propagator.simple_propagation import scalar_wavefronts
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
    with patch.object(scalar_wavefronts, 'apply_compact_index_slice', tracked):
        HologramPipeline(config, HologramPipelineRanges(), cached, 1, verbose=False).run()
    assert calls == list(range(9)) + [8]
    IndependentScalarPipeline(config, HologramPipelineRanges(), reference, 1, verbose=False).run()
    with h5py.File(cached) as a, h5py.File(reference) as b:
        assert a['00000/metadata/propagator/CR/reused_prefix_layers'][()] == 0
        assert a['00000/metadata/propagator/CL/reused_prefix_layers'][()] == 8
        assert 'scalar_prefix_cache' not in a['00000/metadata/propagator_config']
        for pol in ('CR', 'CL'):
            for source in ('exit_wave', 'ideal'):
                np.testing.assert_array_equal(a[f'00000/{pol}/{source}'][()], b[f'00000/{pol}/{source}'][()])
