# Validation record

Executed locally on 2026-09-25 with Python 3.14.7, NumPy 2.5.3,
SciPy 1.18.0 and Matplotlib 3.11.1, using the Agg backend.

- The full `experiments.py --experiment all` pipeline completed.
- Tutorial 17 and all four paper notebooks executed every code cell in order,
  each in a fresh namespace from its own directory. The check used direct Python
  execution of the notebook code cells, not a Jupyter kernel/nbclient runner.
  Notebook source files are intentionally saved without generated cell outputs.
- PNG and PDF figures and raw arrays were generated in `results/`.
- Elastic Ewald geometry, active rotations, the analytic Bragg condition,
  unit magnetization, coverage averaging and undefined missing voxels passed.
- Halving longitudinal slice thickness from 2 to 1 nm: selected Bragg-region
  relative intensity L2 difference **6.82853e-5**.
- Refining transverse pixels from 0.75 to 0.5 nm at fixed 96 nm field of view:
  selected Bragg-region relative intensity L2 difference **3.11157e-4**.
- Normal-incidence weak-medium first-order integrated multislice signal divided
  by the projection-only control: **0.00512428**.
- Tabulated-Co transmitted power fraction at the selected Bragg angle:
  **0.229105** on the validation grid.
- FTH helicity-difference relative L2 against Jones: scalar **8.50269e-7**;
  pure-state Stokes **0**. A separate all-close assertion passed.
- Two repeated seeded detector runs were exactly array-equal.
- `git diff --check` passed.

These checks apply to the declared examples and selected reciprocal regions.
They are not a substitute for the broader validation described in the manuscript.

## Existing test-suite observation

The unchanged Jones test module was run with `unittest`: **31 of 32 passed**.
`test_scalar_matches_jones_for_diagonal_eigenmode` fails its final hologram
comparison at two effectively zero pixels: approximately 4.44e-31 versus
2.47e-31. Its assertion has zero absolute tolerance, so a 1.97e-31 absolute
roundoff difference fails the relative-only comparison. No simulator or existing
test code was changed in this work. The broader pytest selection was unavailable
because this environment has no pytest installation.

The new notebook calculations and physical checks passed independently of that
existing test assertion. Actual run provenance and source/database hashes are
saved in `results/provenance.json`; regenerate them after any code changes.
