"""Random multilayers respect physical constraints and parent reuse."""
import re
import runpy

import h5py
import numpy as np

from scattering_calculator.sample_generator.structures import parse_recipe
from scattering_calculator.simulation_pipelines.recipe_sampling import sample_multilayer_recipe
from scattering_calculator.simulation_pipelines.pipelines.hologram_pipeline import HologramPipeline, HologramPipelineRanges
from scattering_calculator.simulation_pipelines.pipelines.hierarchical_hologram_pipeline import HierarchicalHologramPipeline


def test_sampled_recipe_constraints_and_repeat_expansion():
    rng = np.random.default_rng(5)
    counts, pairs = set(), set()
    for _ in range(300):
        recipe = sample_multilayer_recipe(rng)
        assert all(re.fullmatch(r'\d+\.\d', value)
                   for value in re.findall(r'\(([^)]+)\)', recipe))
        n = int(re.search(r'\]x(\d+)', recipe)[1])
        counts.add(n)
        layers = parse_recipe(recipe).layers
        assert len(layers) == n + 2
        assert all(layer.is_composite for layer in layers[:n])
        gold, chromium = layers[0].components
        assert (gold[0], chromium[0]) == ('Au', 'Cr')
        np.testing.assert_allclose(gold[1] / chromium[1], 4.)
        assert 700 <= sum(layer.thickness for layer in layers[:n])*1e9 <= 3000
        assert layers[n].material == 'SiN'
        assert 50 <= layers[n].thickness*1e9 <= 500
        magnetic = layers[-1]
        assert magnetic.is_composite and len(magnetic.components) == 3
        assert magnetic.components[0][0] == 'Co'
        assert all(2 - 1e-10 <= thickness*1e9 <= 30 + 1e-10 for _, thickness in magnetic.components)
        first, second = magnetic.components[1][0], magnetic.components[2][0]
        assert first != second
        assert {first, second} <= {'Pt', 'Al', 'Ta', 'Ir', 'Cu'}
        pairs.add((first, second))
    assert counts == set(range(7, 16))
    assert len(pairs) == 20
    assert sample_multilayer_recipe(np.random.default_rng(8)) == sample_multilayer_recipe(np.random.default_rng(8))


def test_recipe_varies_between_parents_and_is_saved_for_all_tree_leaves(tmp_path):
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    recipes = iter(['[Au(8)Cr(2)]x7/SiN(20)/Co(6)Ta(2)Ir(3)',
                    '[Au(12)Cr(3)]x8/SiN(25)/Co(8)Cu(3)Al(4)'])
    path = tmp_path/'recipes.h5'
    HierarchicalHologramPipeline(config=factory(), ranges=HologramPipelineRanges(recipe=lambda _: next(recipes)),
        output_path=path, n_samples=2, verbose=False,
        acquisition_variants=[{}, {'measurement_config': {'exposure_time': .2}}]).run()
    with h5py.File(path) as h5:
        saved = [h5[f'{i:05d}/metadata/sample/recipe'].asstr()[()] for i in range(4)]
        assert saved[0] == saved[1]
        assert saved[2] == saved[3]
        assert saved[0] != saved[2]
    pipeline = HologramPipeline(config=factory(), ranges=HologramPipelineRanges(recipe='Co(4)/Al(3)'),
        output_path=tmp_path/'independent.h5', n_samples=1, verbose=False)
    pipeline.run()
    with h5py.File(pipeline.output_path) as h5:
        assert h5['00000/metadata/sample/recipe'].asstr()[()] == 'Co(4)/Al(3)'
