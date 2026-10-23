"""Generate many acquisitions by branching shared magnetic patterns and waves.

Run from the repository root:
    PYTHONPATH=src python tutorials/simulate_hologram_tree.py --patterns 5
"""
import argparse
from pathlib import Path

import numpy as np

from fomocid import DATA_ROOT
from scattering_calculator.simulation_pipelines.recipe_sampling import sample_multilayer_recipe
from scattering_calculator.simulation_pipelines import Uniform

from scattering_calculator.simulation_pipelines.magnetic_parameters import sample_constrained_domain_parameters
from scattering_calculator.simulation_pipelines.pipelines.hologram_pipeline import HologramPipelineConfig, HologramPipelineRanges
from scattering_calculator.simulation_pipelines.pipelines.hierarchical_hologram_pipeline import HierarchicalHologramPipeline


def main():
    """Parse output settings and run the editable example branching tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patterns', type=int, default=5)
    parser.add_argument('--output', type=Path, default=DATA_ROOT / 'Data' / 'hologram_tree' / 'simulation_tree.h5')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fixed-recipe', action='store_true',
                        help='Use config.recipe for every parent instead of random stacks')
    args = parser.parse_args()
    if args.patterns < 1:
        parser.error('--patterns must be positive')
    randomize_recipe = not args.fixed_recipe  # False keeps config.recipe unchanged
    saturation_probability = 0.1  # per magnetic parent, inherited by all its leaves
    rng = np.random.default_rng(args.seed)
    config = HologramPipelineConfig(
        # Fixed recipe used with --fixed-recipe (or randomize_recipe=False).
        recipe='Au(100)/SiN(100)/Co(20)', detector_shape=(256, 256),
        detector_distance=.04, xray_energy=778., xray_photon_flux=1e8,
        propagator_method='Scalar', beamstop_method=None, random_seed=args.seed,
        aperture_radii=[300e-9, 30e-9, 30e-9],
        aperture_centers=[(0., 0.), (1e-6, 0.), (0., 1e-6)],
        pattern_type='constrained_domain_pattern',
        pattern_config={'characteristic_length': 160e-9, 'sigma': 5e-9,
                        'target_magnetization': None, 'max_length_iterations': 1},
    )
    # L sample/energy branches. OH is unchanged; RH size may vary.
    exits = [{}, {'xray_energy': 780.,
                  'aperture_radii': [300e-9, 40e-9, 40e-9]}]
    # Distance is sampled once per independent parent (5–40 cm), not here.
    # Every branch below that parent retains its camera geometry and grid.
    geometries = [{}]
    # N acquisitions per ideal pair. Defaults retain linear detector response
    # and 10 photon-response variants. All sensor artifacts remain available.
    acquisitions = [{'measurement_config': {'exposure_time': exposure,
                                            'number_frames': frames}}
                    for exposure, frames in [(.01, 1), (.1, 1), (.1, 10)]]
    total = args.patterns * len(exits) * len(geometries) * len(acquisitions)
    print(f'{args.patterns} patterns -> {total} CR/CL acquisition pairs')
    HierarchicalHologramPipeline(config=config,
        ranges=HologramPipelineRanges(
            recipe=(lambda _: sample_multilayer_recipe(rng)) if randomize_recipe else None,
            detector_distance=Uniform(.05, .40), pattern_config=lambda _: sample_constrained_domain_parameters(
            rng, saturation_probability=saturation_probability)),
        output_path=args.output, n_samples=args.patterns, exit_variants=exits,
        geometry_variants=geometries, acquisition_variants=acquisitions).run()


if __name__ == '__main__':
    main()
