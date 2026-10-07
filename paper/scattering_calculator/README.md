# Scattering calculator: a normal-incidence FTH walkthrough

Start with one conventional magnetic Fourier-transform holography sample and
follow the light from the incident beam to detector counts. The paper explains
what users configure at each stage, then extends the example to a Co-edge energy
series, selectable interaction/propagation schemes and validation.

- [Manuscript](manuscript.md): the experiment, methods and five-figure narrative.
- [API guide](API_GUIDE.md): runnable calls, options, units and outputs.
- [01 — Basic FTH experiment](notebooks/01_fth_basics.ipynb): sample, exit fields,
  physical detector projection and acquisition saved to HDF5.
- [02 — Hyperspectral series](notebooks/02_hyperspectral.ipynb): optical constants,
  common-q reconstructions and a fixed-camera energy series in the same HDF5 workflow.
- [03 — Interaction and propagation options](notebooks/03_model_options.ipynb).
- [04 — FTH validation](notebooks/04_fth_validation.ipynb).
- [Validation record](VALIDATION.md).

From the repository root, in an environment with NumPy, SciPy, Matplotlib and h5py:

```bash
pip install -r paper/scattering_calculator/requirements.txt
jupyter lab
```

Each primary notebook defines one `ExperimentConfig`, then uses the shared API:

```python
results = sim.simulate_experiment(experiment)
results = sim.load_results(experiment.outputs.path)
figure = sim.plot_results(results)
# Or: results, figure = sim.run_experiment(experiment)
```

The detector is already in the config. `illumination.energies_eV` and
`illumination.polarizations` declare scan axes; `None` uses the source's single
energy/state. `outputs` declares the HDF5 path, saved observables and optional
summary panels. Loading and plotting do not rerun physics. See the
[standard setup guide](../../docs/experiment_setup.md) and
[Tutorial 00](../../tutorials/00_experiment_workflow.ipynb).

`workflow.py` and `experiments.py` retain optional paper-figure and validation
helpers. The older figure generator can still be run with
`MPLBACKEND=Agg python paper/scattering_calculator/workflow.py`, using
`--experiment baseline`, `spectral`, or `validation` and `--output` as before.
Its synthetic display count budget is separate from the production HDF5 runner,
which honors configured flux, exposure and detector response.

## Preserved earlier work

The complete previous project was copied into
[paper/backups/scattering_calculator_2026-10-02](../backups/scattering_calculator_2026-10-02/BACKUP.md)
before rewriting: manuscript, README, validation record, references, notebooks,
experiment implementation, tests and existing local results. A SHA-256 manifest
records every copied file. Generated results stay ignored by Git, so commit the
source backup and retain the local backup directory when archiving the figures.
The original four notebooks are also retained here as earlier research examples;
the numbered walkthrough links above identify the new primary sequence.
