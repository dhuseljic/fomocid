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
        n = int(re.search(r'\]x(\d+)', recipe)[1])
        counts.add(n)
        layers = parse_recipe(recipe).layers
        assert len(layers) == 2*n + 4
        thicknesses = np.array([layer.thickness for layer in layers])*1e9
        np.testing.assert_allclose(thicknesses[:2*n:2] / thicknesses[1:2*n:2], 4.)
        assert 700 <= thicknesses[:2*n].sum() <= 3000
        assert 50 <= thicknesses[-4] <= 500
        assert np.all((thicknesses[-3:] >= 2) & (thicknesses[-3:] <= 30))
        first, second = layers[-2].material, layers[-1].material
        assert first != second
        assert {first, second} <= {'Pt', 'Al', 'Ta', 'Ir', 'Cu'}
        pairs.add((first, second))
    assert counts == set(range(5, 31))
    assert len(pairs) == 20
    assert sample_multilayer_recipe(np.random.default_rng(8)) == sample_multilayer_recipe(np.random.default_rng(8))


def test_recipe_varies_between_parents_and_is_saved_for_all_tree_leaves(tmp_path):
    factory = runpy.run_path('tests/test_constrained_pipeline.py')['small_hologram_config']
    recipes = iter(['Au(10)/SiN(20)/Co(6)/Ta(2)/Ir(3)',
                    'Au(12)/SiN(25)/Co(8)/Cu(3)/Al(4)'])
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
