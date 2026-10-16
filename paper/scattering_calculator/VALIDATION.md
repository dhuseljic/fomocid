# FTH walkthrough validation record

Executed locally on **2026-10-02** with Python 3.10.12, NumPy 1.26.4,
SciPy 1.15.3 and Matplotlib 3.7.0 (Agg backend). Actual runtime versions and
source/database hashes are saved in generated provenance files; inspect those
files for the precise environment used by a later run.

## Executed checks

- `workflow.py` completed baseline, spectral and validation calculations.
- All code cells in the four new notebooks executed in order from their notebook
  directory, each with a fresh namespace. This used direct Python execution,
  not a Jupyter kernel runner. Source notebooks retain no generated outputs.
- The model-options notebook ran Fraunhofer and direct Rayleigh–Sommerfeld on
  the same Jones exit fields with a 24 × 24 detector.
- The energy series saved an 8 × 192 × 192 common-q complex reconstruction cube
  and an 8 × 2 × 128 × 128 fixed-camera helicity stack with per-energy q grids.
- Jones/pure-state Stokes differences were at approximately 3e-13 in relative L2
  norm; the asserted all-close consistency check passed.
- Scalar/Jones helicity-difference discrepancy was approximately **8.51e-7**.
- Halving the maximum slice thickness from 5 to 2.5 nm changed the helicity
  difference by **1.90e-5** in relative L2 norm.
- Omitting within-sample transverse propagation changed that difference by
  **0.1233** relative to the multislice prediction. This is model sensitivity,
  not numerical error.
- Doubling transverse pixels while halving pitch at fixed field of view changed
  the common-q helicity difference by **0.05009** in relative L2 norm. The fine
  orthonormal FFT intensity was divided by four before comparison on the common
  q support. The starter grid therefore is not established as converged for
  quantitative FTH imaging; the small reference aperture needs refinement.
- Physical-detector intensities were finite and nonnegative. Repeated seeded
  acquisition arrays were exactly equal.
- `tests/test_detector_propagation.py`: **23 passed**, covering independent
  analytic aperture, angular-spectrum, adjoint, far-field convention and detector
  integration checks. One pre-existing SciPy import deprecation warning appeared.
- The backup manifest verified all **40** copied files, including original local
  results. Generated figure outputs were inspected visually.

Reproduce the production checks from the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src MPLBACKEND=Agg \
  python -m pytest tests/test_detector_propagation.py -q
```

Plugin autoload was disabled because an unrelated installed napari/Numba pytest
plugin could not initialize its cache under the filesystem permissions. The
figure requirements also now include h5py, ipywidgets and tqdm, which are needed
by the simulator's package imports in a fresh environment.

## Scope and remaining work

These calculations check the stated normal-incidence sample and selected code
paths. They do not establish experimental accuracy, mixed-polarization accuracy,
complete vector interface physics or calibrated absolute photon counts. Refine
transverse sampling, source extent, longitudinal slices and detector footprints
for the observable of interest. Match backend coordinate conventions and
normalization before quantitative Fraunhofer/RS comparisons.

The earlier skyrmion validation record and its known near-zero-pixel Jones test
observation are preserved unchanged in the
[dated backup](../backups/scattering_calculator_2026-10-02/VALIDATION.md).


## Dataset-driven optical response and detector calibration — 15 October 2026

The optional optical response interface now takes channel selectors and spatial
axis/strength fields only. All coefficients come from the common material
refractive-index datasets; the earlier inline-spectrum prototype has been removed.
No natural or ferroelectric datasets are bundled, and missing enabled channels
raise explicit errors. Isolated test directories supply synthetic material tables
for numerical checks; those coefficients are not package or notebook defaults.

Executed checks cover:

- Original empty/default material behavior and a zero added material channel on
  a masked FTH specimen, retaining its magnetic support treatment.
- Independent matrix-exponential transmission, axis reversal, circular
  handedness/reciprocity, vacuum masking, nominal/local projection and scalar
  eigenmodes. Named magnetic linear tables use the same response loader.
- Material-table interpolation and range failures, per-energy and per-color
  refresh, and recorded evaluated coefficients/source hashes in HDF5.
- Mixed Stokes tilt/defocus with transmitted padding backgrounds against
  independent Jones carriers.
- Physical Poisson mean/variance with gain applied afterward, constant/callable
  gain equivalence, independently sampled simultaneous colors, one shared
  nonlinear signal conversion, and per-frame clipping before averaging.
- Coordinate-energy overrides leave physical gain evaluation unchanged.
  Hyperspectral/harmonic acquisitions record evaluated gains and reproduce
  seeded outputs. Invalid calibration values and shapes fail explicitly.
- Both updated notebooks executed all default cells. All optional illumination
  and detector controls executed together using a small camera, three frames
  and 50 nm maximum slices. The enabled optical notebook executed against
  isolated test material files with 50 nm maximum slices. These are smoke checks,
  not convergence results or calibration of a real camera/material.

The focused material/propagation/detector suite passed **198 tests**. The same
pre-existing near-zero Jones FFT assertion remains excluded; it was reproduced
on the original propagator code during the preceding refactor. Notebook schemas
and source cells were validated. The paper and guides distinguish calibrated
nonlinear response from spatial/temporal sensor effects such as brighter-fatter
charge redistribution, blooming or persistence, which are not claimed here.

Physical acquisitions with nonunit ADU/photon now have the correct gain-scaled
mean and variance; the older divide-before-Poisson conversion canceled gain from
the mean. Multiple frames now undergo acquisition and clipping independently,
then are averaged. These corrections intentionally change affected detector
counts. Constant gain, linear response and disabled additional optical channels
remain the defaults.
