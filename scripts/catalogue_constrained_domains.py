"""Measure experimental constrained dipolar domains; no optical simulation."""
from pathlib import Path
import csv
import json
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scattering_calculator.sample_generator import create_constrained_domain_pattern as generate


def main():
    destination=Path('docs/assets/constrained_domains')
    destination.mkdir(parents=True,exist_ok=True)
    seeds=[7,11,23]
    records=[]
    # Main scan separates occupancy, directional energy bias and length.
    for length,alignment,magnetization in itertools.product([12.,16.,24.],[0.,8.],[-.6,0.,.6]):
        for seed in seeds:
            params=dict(characteristic_length=length,target_magnetization=magnetization,
                        alignment=alignment,seed=seed,n_steps=400)
            _,metadata=generate((128,128),**params)
            records.append(dict(parameters=params,metadata=metadata))
        print(f'length={length}, alignment={alignment}, M={magnetization}: measured',flush=True)
    cases=[('Labyrinth',dict()),('Aligned stripes',dict(alignment=8.)),
           ('Minority bubbles',dict(target_magnetization=.6)),
           ('Ordered bubble seed',dict(target_magnetization=-.6,initial_state='hexagonal',initial_order=1.)),
           ('Ordered stripe seed',dict(initial_state='stripes',initial_order=1.,alignment=8.)),
           ('Pinned labyrinth',dict(disorder=.5,disorder_length=1.))]
    fig,axes=plt.subplots(2,3,figsize=(12,9),layout='constrained')
    for ax,(label,controls) in zip(axes.ravel(),cases):
        pattern,metadata=generate((128,128),characteristic_length=16.,n_steps=400,seed=7,**controls)
        records.append(dict(label=label,parameters=dict(characteristic_length=16.,n_steps=400,seed=7,**controls),metadata=metadata))
        ax.imshow(pattern,cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
        ax.set_axis_off()
        length=metadata['measured_length_px']
        ax.set_title(f"{label}\nM={metadata['mean_mz']:+.3f}; L={length:.1f}px; {metadata['elapsed_seconds']:.2f}s\nlength target met={metadata['length_target_met']}; relaxed={metadata['converged']}",fontsize=9)
    fig.suptitle('Experimental constrained dipolar model: fully magnetized +/-1 domains\n128×128, target period 16 px, seed 7; up to 400 steps per calibration attempt',fontsize=12)
    fig.savefig(destination/'morphologies.png',dpi=140)
    plt.close(fig)
    data=dict(schema_version=1,shape=[128,128],seeds=seeds,records=records)
    (destination/'measurements.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    rows=[]
    for r in records:
        m=r['metadata']
        rows.append(dict(label=r.get('label','scan'),**r['parameters'],
                    achieved_magnetization=m['mean_mz'],measured_period=m['measured_length_px'],
                    length_target_met=m['length_target_met'],length_relative_error=m['length_relative_error'],
                    nematic_order=m['nematic_order'],converged=m['converged'],residual=m['residual'],
                    steps_completed=m['steps_completed'],elapsed_seconds=m['elapsed_seconds'],
                    calibration_attempts=len(m['calibration_history']),
                    bubbles=m['measurements']['bubble_count'],stripes=m['measurements']['stripe_count']))
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with (destination/'measurements.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    print(f'Saved {len(records)} records',flush=True)


if __name__=='__main__': main()
