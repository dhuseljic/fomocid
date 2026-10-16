"""Generate the measured parameter atlas without running optical simulations.

Run from the repository root:
PYTHONPATH=src python scripts/catalogue_magnetic_domains.py
"""
from pathlib import Path
import argparse
import csv
import itertools
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scattering_calculator.sample_generator.pattern_generator import create_domain_pattern
from scattering_calculator.sample_generator.domain_catalogue import measure_domain_pattern


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/assets/domain_catalogue'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = dict(period=16., k0=.7, eps=.8, n_steps=80, noise_amp=0.,
                use_gpu=False, sigma=.7, domain_conversion="hard")
    panels = [
        ('occupancy_order', 'up_fraction', [.15, .3, .5, .7, .85], 'lattice_order', [0., .5, 1.], {}),
        ('evolution_bias', 'target_mean', [-.8, -.4, 0., .4, .8], 'quadratic_coefficient', [-.4, 0., .4], {}),
        ('length_dynamics', 'eps', [-.8, -.4, 0., .4, .8], 'k0', [.5, .7, 1.1], {'up_fraction':.3, 'lattice_order':.5}),
        ('native_stripes', 'eps', [.1, .35, .6, .9, 1.2], 'k0', [.15, .4, .7],
         {'target_mean':0., 'up_fraction':None, 'lattice_order':0.,
          'stable_evolution':False, 'rescale_mode':'native', 'sigma':None, 'max_hole_area':9}),
    ]
    records = []
    for name, xkey, xs, ykey, ys, fixed in panels:
        fig, axes = plt.subplots(len(ys), len(xs), figsize=(14, 9), squeeze=False, layout='constrained')
        for (row, y), (col, x) in itertools.product(enumerate(ys), enumerate(xs)):
            cell_records = []
            for seed in (11, 23, 47):
                parameters = dict(base, **fixed)
                parameters.update({xkey:x, ykey:y, 'seed':seed})
                pattern, metadata = create_domain_pattern((128,128), **parameters)
                metrics = measure_domain_pattern(pattern)
                record = dict(panel=name, row=row, column=col, seed=seed,
                              parameters=parameters, measurements=metrics, generator_metadata=metadata)
                records.append(record);cell_records.append(record)
                if seed == 11:
                    axes[row,col].imshow(pattern, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
                    axes[row,col].set_axis_off()
            fractions = [r['measurements']['up_fraction'] for r in cell_records]
            labels = ','.join(sorted({r['measurements']['morphology'] for r in cell_records}))
            axes[row,col].set_title(f'{xkey}={x:g}; {ykey}={y:g}\nf+={np.mean(fractions):.2f} ± {np.std(fractions):.2f}; {labels}', fontsize=8)
        scale_label = 'native solver scale (period follows k0)' if name == 'native_stripes' else 'nominal period 16 px'
        fig.suptitle(f'{name}: images seed 11; measurements across seeds 11, 23, 47\n128×128, {scale_label}, 80 steps; synthetic scalar magnetization', fontsize=12)
        fig.savefig(args.output / f'{name}.png', dpi=120)
        plt.close(fig)
        print(f'{name}: {len(xs)*len(ys)*3} measured samples', flush=True)
    def clean(value):
        if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value, list): return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
        return value
    (args.output/'catalogue.json').write_text(json.dumps(clean(dict(
        schema_version=1, shape=[128,128], seeds=[11,23,47], records=records)), indent=2, allow_nan=False)+'\n')
    flat = [dict(panel=r['panel'], row=r['row'], column=r['column'],
                 **{f'parameter_{k}':v for k,v in r['parameters'].items()},
                 **r['measurements']) for r in records]
    columns = list(dict.fromkeys(k for row in flat for k in row))
    with (args.output/'catalogue.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=columns);writer.writeheader();writer.writerows(flat)


if __name__ == '__main__':
    main()
