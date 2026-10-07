# A practical guide to simulating magnetic Fourier-transform holography

**Software-methods manuscript draft.** Authorship, affiliations, funding and an archived release identifier remain to be supplied.

## Abstract

The scattering calculator in fo-mo-cid models the steps between a specified sample and a recorded coherent X-ray image. We demonstrate its use with a normal-incidence Fourier-transform holography (FTH) specimen: a patterned absorbing mask, a magnetic multilayer and a reference aperture. The example follows sample construction and energy-dependent optical constants, local light–matter interaction, propagation within the sample, free-space propagation to a physical detector, and detector response. Configuration examples show how users select field representations and propagation approximations. An energy series across the cobalt absorption-edge region illustrates hyperspectral imaging, followed by comparisons of numerical schemes and numerical validation. The purpose is to make the forward model usable and its assumptions inspectable.

## 1. One experiment, from sample to camera

FTH records interference between light transmitted by an object aperture and a smaller reference aperture. Our starting point is a coherent beam along +z, perpendicular to a planar sample, with matched right- and left-circular illumination (CR and CL). Figure 1 pairs a sketch of this experiment with the operations performed by the code. Users specify the specimen and source, compute complex exit fields, propagate these fields to the detector, and apply an acquisition model. Reconstruction is a subsequent analysis of the simulated measurement.

The reference specimen is Au(300)/SiN(20)/Pt(2)/Co(10)/Pt(2), with thicknesses in nm. An object hole of radius 180 nm removes the Au while retaining the underlying magnetic film. A 30 nm radius reference hole, displaced by (600, 0) nm, passes through the entire stack. A prescribed domain texture supplies the Co magnetization. It is an input to the optical calculation, not a predicted equilibrium magnetic state. The reference separation exceeds 3R_object + R_reference = 570 nm, separating the ideal reference cross correlation from the central autocorrelation.

The baseline uses 778 eV, a 192 × 192 sample grid at 10 nm pitch, and a Gaussian amplitude envelope with width 650 nm. Each material is divided into slices no thicker than 5 nm; this maximum is a numerical parameter, not a claim of convergence. The flat detector has 128 × 128 pixels of 13.5 µm pitch at 5 cm. These are illustrative settings rather than a calibrated beamline. Figure 2 shows the sample, exit intensity, ideal detector image and a synthetic detector realization. All settings can be edited in the first notebook.

![Figure 1: experiment and workflow](results/fth_workflow/fig01_setup_workflow.png)

## 2. How users specify the forward model

The executable teaching interface is [workflow.py](workflow.py); it calls the production material, scalar/Jones/Stokes propagation, detector-projection and acquisition kernels. The existing [experiments.py](experiments.py) supplies the shared FTH sample builder. [API_GUIDE.md](API_GUIDE.md) gives runnable calls, units, returned arrays and the corresponding production configuration selectors.

### 2.1 Sample design and optical constants

`experiment.sample.recipe` defines explicit material interfaces. Object and reference radii and the reference offset determine masks, which act on material occupancy in each slice. `experiment.simulation.shape`, `real_space_pixel_size` and `experiment.sample.max_slice_thickness` set the transverse grid and longitudinal discretization. Magnetization is a three-component array. The shared `ScatteringExperiment.setup(magnetization=..., mask=...)` accepts externally supplied arrays; aperture configuration controls hole penetration. All maintained examples follow the same [experiment setup](../../docs/experiment_setup.md).

For each photon energy, `material_params` loads the charge, circular and linear optical channels `(n0, nc, nl)`. The convention is n0 = 1 − δ − iβ with negative spatial propagation phases, so positive β produces attenuation. The bundled Co magnetic channel is available in the loader's 770–805 eV window. Generic material loading supplies zero circular and linear channels, and the linear channel in this Co example is zero. Users must check channel availability; specifying a magnetic element alone does not provide its dichroic spectrum.

Explicit layers and thickness-weighted effective mixtures answer different questions. The recipe interface uses `/` to separate layers; adjacent material terms can specify an effective layer. Use explicit interfaces when within-sample propagation through the layer arrangement matters. Optical-table provenance and redistribution permissions still need to be established for publication.

### 2.2 Local interaction and within-sample propagation

`mode="Scalar"`, `"Jones"` or `"Stokes"` selects the field representation. Scalar carries one complex field for the chosen polarization response. Jones carries two coherent transverse components and permits polarization conversion in the implemented material model. Stokes exposes `[I,Q,U,V]` while retaining coherent Jones carriers internally; mixed polarization modes are combined incoherently. The four real Stokes components are not Fourier-transformed as wave amplitudes.

For each slice, local material transmission applies absorption and phase delay. Scalar transmission is exp(−ik n Δz); Jones transmission uses the matrix square root and exponential of the projected dielectric tensor. With `propagate=True`, the code also advances the field between slices with the angular-spectrum kernel exp(−i kz Δz), where kz² = k² − kx² − ky² for propagating frequencies. Evanescent contributions decay. The default FTH walkthrough explicitly enables this multislice propagation.

With `propagate=False`, local material interactions remain active but transverse diffraction between slices is omitted. This provides a useful projection-style control. It does not disable exit-to-detector propagation. The production `SamplePropagatorConfig` exposes the same choice as `propagator_config={"propagate": True}` and the field representation as `propagator_method="Jones"`. The two controls should be varied separately.

### 2.3 Free-space propagation and projection onto detector pixels

`experiment.detector.detector_propagation_method="fraunhofer"` computes the ideal reciprocal intensity by FFT and projects it onto physical flat-detector pixels. Angular mapping and a relative flat-pixel solid-angle factor are retained. `detector_pixel_footprint_samples=3` averages nine sample points per pixel rather than using pixel centres alone. Detector distance and pitch control angular acceptance independently of the sample grid.

`method="rayleigh_sommerfeld"` evaluates finite-distance scalar diffraction directly at detector coordinates, propagating each coherent channel before adding its intensity. This option changes only the exit-to-detector step. It can be expensive because direct integration couples source and detector pixels. The kernel uses zero exterior field outside the supplied sample window, so source-window and sampling convergence matter. A two-component input does not turn this scalar free-space operator into a full vector Maxwell boundary calculation.

In the production API, these options are `DetectorConfig.detector_propagation_method`, `use_detector_pixel_footprint` and `detector_pixel_footprint_samples`. `ignore_flat_detector_curvature=True` selects linear small-angle detector mapping for Fraunhofer; leave it False for Rayleigh–Sommerfeld. See the [propagation guide](../../docs/light_propagation_modes.md) for the numerical conventions. The legacy FFT and RS paths have a documented transverse-coordinate reversal, and their raw normalization differs in the teaching helper. Align coordinates and establish a common physical normalization before making quantitative backend comparisons.

### 2.4 Acquisition and detector artifacts

The walkthrough assigns one multiplicative synthetic photon budget to the two ideal helicity images together. Separate peak normalization of CR and CL would distort their difference. The standard `experiment.detector` section controls exposure, frame count, efficiency, readout noise, saturation, photon spreading, hot/cold pixels and cosmic rays. A central beamstop mask is applied in detector pixels. This simple teaching mask does not model a beamstop wire or its diffraction; the production `BeamstopConfig` offers additional shadow geometry controls.

The baseline illustrates a peak budget of 20,000 before acquisition scaling, unit exposure and efficiency, two-count readout noise and saturation at 16,000 counts. Camera and exposure seeds allow repeated realizations. The raw ideal intensity, expected synthetic budget, measured counts and beamstop mask are saved separately. Absolute photon-flux calibration is outside this example.

## 3. From a single hologram to hyperspectral imaging

The second notebook repeats the same specimen at 772, 775, 778, 780, 783, 790, 795 and 800 eV. Optical constants and wavelength are reloaded at each energy; geometry and the incident amplitude envelope remain fixed. These sparse points demonstrate the workflow across the Co edge region. Users can replace the energy list with a finer scan within the available table window.

Two complementary data products are saved. `hyperspectral.npz` contains CR−CL holograms and complex FTH reconstructions on a common transverse-q grid. `fixed_detector_series.npz` contains both helicity images for an unchanged physical detector, together with qx(E) and qy(E). Wavelength changes the q coordinates of stationary detector pixels. Consequently, a fixed-camera stack must be regridded before a common-q reconstruction or spatial comparison; the common-q demonstration cube is computed before physical detector projection.

Figure 3 illustrates the common-q FTH reconstruction. An inverse Fourier transform of the helicity difference gives reference cross correlations, and the selected sideband is blurred by the finite reference hole. Its magnitude discards sign and phase; it is not a calibrated vector magnetization map. Figure 4 shows the circular optical channels, a sideband-amplitude spectrum and an energy–position section. The sideband metric in the current helper is the root-sum-square amplitude over the ROI (historically saved as `sideband_rms`); absorption, phase and interference all contribute. It must not be identified with an absorption coefficient.

## 4. Compare interaction and propagation schemes

Figure 5 compares Scalar, Jones and pure-state Stokes calculations on the same specimen and q grid. This normal-incidence example has a zero linear optical channel, making circular polarization a useful scalar eigenmode control. Pure-state Stokes shares its Jones carrier, so agreement checks implementation consistency. It is not an independent validation of the electromagnetic model or a guarantee of scalar accuracy for tilted or anisotropic specimens.

The third notebook separately compares `propagate=True` with `False` and reports the relative intensity-difference norm. It then demonstrates switching between Fraunhofer and Rayleigh–Sommerfeld for the same exit fields, using a small detector to keep direct integration manageable. Each backend is displayed with a stated relative scale. An image-shape or brightness difference alone cannot be attributed to finite-distance physics until coordinate conventions, normalization, transverse support and pixel sampling have been matched.

## 5. Validation and limits

The fourth notebook checks finite nonnegative detector intensity, repeatability of seeded acquisition and Jones/pure-state-Stokes agreement. It halves the maximum material-slice thickness and reports the change in the CR−CL hologram. That number is specific to this specimen and grid; it is a refinement observation rather than a universal error bound. Production tests independently check the RS operator against an analytic circular-aperture result, a padded angular-spectrum calculation, its discrete adjoint and detector integration.

Before using a prediction quantitatively, refine transverse pitch at fixed field of view, enlarge the source window, refine longitudinal slices and detector footprints, and inspect convergence of the actual observable of interest. The 30 nm reference aperture is only three sample pixels in radius at the baseline pitch and particularly needs refinement. Check optical-channel availability and use matched geometry, illumination and normalization in each comparison. Numerical convergence cannot establish the accuracy of tabulated material response, assumed magnetic textures or an uncalibrated camera model.

The model is a forward transverse-field calculation. General backscattering and full electromagnetic interface boundary conditions are omitted. No experimental accuracy claim is made here. Executed checks and remaining work are recorded in [VALIDATION.md](VALIDATION.md).

## 6. Reproducibility and figure plan

Run `MPLBACKEND=Agg python paper/scattering_calculator/workflow.py` from the repository root to regenerate the main FTH figures and raw arrays. The notebooks expose the same functions and editable settings. Generated files go into `results/fth_workflow/`; parameter and source/database provenance accompany the baseline. The previous skyrmion-led draft, notebooks, scripts and local results are preserved in [the dated backup](../backups/scattering_calculator_2026-10-02/BACKUP.md).

| Figure | Purpose | Generated file |
| --- | --- | --- |
| 1 | Normal-incidence experiment sketch and code flow with user controls | `fig01_setup_workflow.pdf` |
| 2 | Sample → exit intensity → physical detector → artifacts | `fig02_fth_chain.pdf` |
| 3 | Common-q magnetic hologram and finite-reference FTH sideband | `fig03_fth_reconstruction.pdf` |
| 4 | Co optical channels and hyperspectral reconstruction series | `fig04_hyperspectral.pdf` |
| 5 | Matched Scalar, Jones and pure-state Stokes comparison | `fig05_interaction_modes.pdf` |

Further propagation comparisons and refinement metrics are interactive notebook outputs and JSON records. The retained [references.bib](references.bib) supplies literature for subsequent citation editing. This draft needs finalized citations, optical-data licensing, author metadata, numerical convergence and an archived release before submission.
