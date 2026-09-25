# Scattering-calculator paper and reproducible figures

This folder contains a software-methods manuscript draft, executable figure notebooks,
and a shared experiment script. It is a starting point for a modest software paper,
not a submitted or experimentally validated publication. The contribution is the
integration and reproducibility of established methods, not a new scattering theory.

- [Manuscript](manuscript.md)
- [Skyrmion tutorial](../../tutorials/17_skyrmion_lattice_ewald_rods.ipynb)
- [Figures 1–4: skyrmion tilt series and reciprocal-space assembly](notebooks/01_skyrmion_rods.ipynb)
- [Figures 5–7: Pt/Co FTH, spectroscopy, and propagation modes](notebooks/02_fth_spectroscopy_modes.ipynb)
- [Figure 8: detector artifacts](notebooks/03_detector_artifacts.ipynb)
- [Numerical validation](notebooks/04_validation.ipynb)
- [Experiment implementation](experiments.py)
- [References](references.bib)
- [Executed checks and known test-suite observation](VALIDATION.md)

From the repository root, using a Python environment with NumPy, SciPy and Matplotlib:

```bash
python paper/scattering_calculator/experiments.py --experiment all
```

For a new environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r paper/scattering_calculator/requirements.txt
python paper/scattering_calculator/experiments.py --experiment all
jupyter lab
```

The helper imports this checkout's simulator directly. No package installation or SSL
training dependencies are needed. Use `MPLBACKEND=Agg` for headless execution.
Open the notebooks from anywhere inside this repository. Each notebook calls the
same script functions, so figures do not depend on hidden state from another notebook.
The validation notebook additionally checks lateral refinement.

The complete script generates PNG and PDF figures, raw NPZ arrays, numeric JSON
reports and source/database hashes in `results/` (ignored by Git). Allow a few minutes
and roughly 2 GB of available memory for the default explicit 3D FFT; actual timings depend on the machine. Use
`--experiment skyrmion`, `multislice`, `fft`, `fth`, `detector`, or `validation` to run a subset;
`--output /path/to/results` changes the destination. Later runs replace matching files.
Each run records the current commit, dirty working-tree status and SHA-256 hashes;
freeze a clean revision and environment before final publication. The requirements
specify compatible minimum versions; `results/provenance.json` records actual versions.

The main skyrmion design is 778 eV, 12 nm lattice spacing, 6.8 nm full texture diameter,
and 269.4766 nm thickness. The first-order rod centres are at the second thickness zero
at normal incidence. Finite rods and finite illumination still produce residual signal.
The figure set separates the Born reference, production weak-medium multislice,
and tabulated-Co multislice. Both Born and multislice scans are mapped onto rotated
Ewald spheres with explicit missing-data coverage. This is reciprocal intensity
assembly, not a recovered three-dimensional magnetization.

Before submission, complete the explicitly listed validation and metadata tasks in
the manuscript. In particular, there is no experimental comparison, full-vector tilted
benchmark, finalized author list, optical-data licensing audit, or archived release DOI.

## Change the experiment in the notebook

Each notebook now has an **Editable experiment parameters** or **Editable sample
and geometry** cell directly below its imports. Edit that cell and Run All;
there is no need to edit `experiments.py`.

- `SkyrmionConfig`: radius, lattice spacing, energy, thickness, displayed tilts,
  both tilt scans, Gaussian beam width, detector pixel count/pitch/distance and
  centre offsets, multislice grids, reciprocal bins, and 3D FFT sampling.
- `FTHConfig`: layer materials/thicknesses, slice thickness, aperture radii and
  reference position, beam width, domains, energy scan and real-space sampling.
- `DetectorEffectsConfig`: count budget, beamstop radius, exposure/frame count,
  readout noise, efficiency, saturation, sensor artifacts and random seeds.

All angles are sample rotations about lab y measured from normal incidence.
All specimen lengths are nm; detector pitch/distance explicitly use metres.
`thickness_nm=None` automatically chooses the second longitudinal zero; enter a
number to keep thickness fixed when changing energy or lattice spacing. The
printed geometry reports whether that choice actually misses the central rod
lobe. The selected angle lists remain exactly what you entered; an optional
notebook snippet derives them from the new Bragg angle.

Notebook outputs use named subfolders of `results/`, and the effective parameters
are saved in provenance. Change `OUT` to retain different parameter runs. The
standalone CLI still writes into `results/` by default. Numbers in the prose and
manuscript describe reference examples, not every custom parameter choice.

## Direct 3D FFT versus diffraction assembly

The skyrmion notebooks also generate:

- `fig09_fft_volume_comparison`: full direct 3D FFT, diffraction assembly,
  Ewald-sampled FFT, matched residual, coverage and a longitudinal rod profile.
- `fig09b_fft_volume_3d`: all three reciprocal-space volumes in matching 3D views.
- `fig09c_multislice_fft_comparison`: production multislice versus the FFT on the
  same detector/scan, with one explicitly reported fitted intensity scale.

The comparison explicitly constructs a voxelized 3D Gaussian-weighted magnetic
contrast image and applies `scipy.fft.fftn` to each component. It uses the same
radius, lattice and slab thickness as the diffraction calculation. `fft_nz`
controls vacuum padding and qz interpolation; `fft_dz_nm` controls actual z-voxel
sampling. The transverse FFT grid follows `born_n` and `born_dx_nm`.

The full scalar volume is `|F[mz−1]|²`. In the default `xmcd` mode, the matched
reference projects complex vector amplitudes onto the angle-dependent incident
direction before squaring. Select `contrast_channel='mz'` for an orientation-
independent scalar control. Coverage remains separate and unmeasured voxels stay
NaN. The FFT comparison does not invent diffraction data in unmeasured regions.
Only a multislice archive with exactly matching saved parameters is used.

This is a Born-limit reference, not an assertion that a full multislice intensity
must equal an object's Fourier intensity. The Gaussian beam is lab-fixed in the
multislice simulation and sample-fixed in the reference; interaction, refraction,
and illumination differences can matter at finite tilt.

Run the configuration/FFT checks with:

```bash
python -m unittest discover -s paper/scattering_calculator -p 'test_*.py'
```
