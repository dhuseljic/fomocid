# Scattering calculator: a normal-incidence FTH walkthrough

Start with one conventional magnetic Fourier-transform holography sample and
follow the light from the incident beam to detector counts. The paper explains
what users configure at each stage, then extends the example to a Co-edge energy
series, selectable interaction/propagation schemes and validation.

- [Manuscript](manuscript.md): the experiment, methods and five-figure narrative.
- [API guide](API_GUIDE.md): runnable calls, options, units and outputs.
- [01 — Basic FTH experiment](notebooks/01_fth_basics.ipynb): sample, exit fields,
  physical detector projection and acquisition; generates Figures 1–2.
- [02 — Hyperspectral series](notebooks/02_hyperspectral.ipynb): optical constants,
  common-q reconstructions and a fixed-camera energy series; Figures 3–5.
- [03 — Interaction and propagation options](notebooks/03_model_options.ipynb).
- [04 — FTH validation](notebooks/04_fth_validation.ipynb).
- [Validation record](VALIDATION.md).

From the repository root, in an environment with NumPy, SciPy and Matplotlib:

```bash
pip install -r paper/scattering_calculator/requirements.txt
MPLBACKEND=Agg python paper/scattering_calculator/workflow.py
jupyter lab
```

Use `--experiment baseline`, `spectral`, or `validation` for a subset, and
`--output /path/to/results` for a different destination. The default is
`results/fth_workflow/`. Runs replace matching outputs; change the destination to
retain parameter sweeps. Each notebook has an editable configuration cell and
can be run independently from any directory inside the checkout.

`workflow.py` is the teaching interface; `experiments.py` retains the shared sample
builder and earlier research helpers. No core simulator code is changed by this
paper revision. The baseline detector count budget is synthetic, not calibrated
beamline flux. A common-q FFT reconstruction and a fixed physical detector are
separate products, documented in the API guide.

## Preserved earlier work

The complete previous project was copied into
[paper/backups/scattering_calculator_2026-10-02](../backups/scattering_calculator_2026-10-02/BACKUP.md)
before rewriting: manuscript, README, validation record, references, notebooks,
experiment implementation, tests and existing local results. A SHA-256 manifest
records every copied file. Generated results stay ignored by Git, so commit the
source backup and retain the local backup directory when archiving the figures.
The original four notebooks are also retained here as earlier research examples;
the numbered walkthrough links above identify the new primary sequence.
