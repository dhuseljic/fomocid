# A modular forward simulator for coherent magnetic X-ray scattering and Fourier-transform holography

**Software-methods manuscript draft — author names, affiliations, contributions, funding, and release identifier to be supplied by the authors.**

## Abstract

Interpreting coherent magnetic X-ray images requires accounting for material contrast, propagation through a specimen, diffraction to a detector, and the measurement process. We present the coherent-scattering calculator distributed within the fo-mo-cid repository, a Python implementation that combines these stages in configurable forward simulations. The software supports scalar and Jones-field propagation, a Mueller–Stokes interface retaining coherent carrier fields, multilayer material recipes, magnetic textures, aperture masks, and detector artifacts. Reproducible examples describe Fourier-transform holography of a Pt/Co film, energy-dependent magnetic contrast, and the angle-dependent diffraction of a prescribed hexagonal skyrmion-tube lattice. A deliberately thick, small-period lattice demonstrates why the projected-object approximation can miss suppression by longitudinal interference. Tilted diffraction patterns are assembled in sample-frame reciprocal space using their Ewald-sphere coordinates and an explicit coverage map. These examples establish a workflow for controlled numerical studies and synthetic data generation; they do not constitute experimental validation or a full electromagnetic treatment of magnetic scattering.

## 1. Motivation and scope

A measured coherent-scattering image is the result of several coupled operations. The magnetization and chemical structure determine a complex, polarization-dependent optical response. Propagation through a patterned mask and a multilayer can alter the field before it reaches the sample exit. Free-space propagation converts this field into an interference pattern. Finite acceptance, photon statistics and detector defects then modify the recorded signal. Keeping these operations explicit makes it possible to ask whether a feature originates in the specimen, propagation, or acquisition.

Magnetic Fourier-transform holography (FTH) is a useful setting for such calculations because a reference aperture provides interference with a magnetic object. Its experimental basis is established by [Eisebitt et al.](https://doi.org/10.1038/nature03139). Here, the objective is a reproducible forward model for studying this measurement chain, rather than a new reconstruction algorithm. The calculator also supports synthetic datasets for method development within the larger fo-mo-cid repository. Data-driven applications require care: matching simulated and measured nuisance distributions is a separate validation task.

The contribution is an accessible integration of material response, established multislice propagation, selectable detector propagation and acquisition effects. Multislice X-ray calculations are established, including applications to nanofocusing optics described by [Li, Wojcik and Jacobsen](https://doi.org/10.1364/OE.25.001831). The present examples emphasize magnetic contrast, masks and reproducible parameter comparisons. We make no claim that multislice, FTH, or Ewald-sphere assembly is novel.

## 2. Computational model

### 2.1 Specimen and optical response

A specimen consists of material layers, three-component magnetization arrays, and masks describing material occupancy. Layer recipes permit explicit interfaces or effective mixtures. In the recipe syntax, `/` separates layers, whereas adjacent material terms form a thickness-weighted effective layer; these representations are not interchangeable when interface propagation matters. Apertures include object and reference holes, and the broader interface supports tapered holes, slits and roughness controls. Synthetic magnetic patterns and externally supplied micromagnetic volumes can be used.

The database loader returns charge, circular and linear channels, represented as $n_0$, $n_c$, and $n_l$. The adopted convention is $n_0=1-\delta-i\beta$, paired with propagation phases of negative sign. A positive $\beta$ therefore attenuates a wave advanced with $\exp(-ikn_0\Delta z)$. The polarization-dependent material response is assembled into local scalar indices or projected dielectric tensors. The physical motivation for magnetization-sensitive resonant contrast is established by [Hannon et al.](https://doi.org/10.1103/PhysRevLett.61.1245); the implemented optical-index model should not be equated with a complete microscopic resonant-scattering calculation.

A material-specific limitation is important for interpreting the examples. The current loader supplies measured-style circular channels for Co only within its special 770–805 eV window, with linear interpolation of the bundled tables. The generic material-loading branch returns zero circular and linear channels; the linear channel is also zero in the Co examples. Consequently, naming a material does not imply that its magnetic dichroism is available, and these demonstrations do not validate XMLD spectroscopy. Optical-constant provenance, calibration and redistribution permissions must be established for a release used in a publication.

### 2.2 Local interaction and multislice propagation

The field advances through local material transmission followed, optionally, by transverse free-space propagation between slices. For scalar response,

$$E_j^+(x,y)=\exp[-ik n_j(x,y)\Delta z_j]E_j^-(x,y).$$

Jones propagation carries two complex transverse components. Its local transmission is computed from the projected dielectric tensor using the matrix square root and exponential. The repository retains a zero-order vacuum phase convention that also operates when transverse inter-slice propagation is disabled. Spatially uniform phase factors do not affect the intensity comparisons here.

Between slices, the angular-spectrum step has the form

$$\mathcal P_{\Delta z}E=\mathcal F^{-1}\left[H(k_x,k_y;\Delta z)\mathcal F(E)\right],\qquad
H=\exp[-i\sqrt{k^2-k_x^2-k_y^2}\,\Delta z].$$

Propagating spatial frequencies use the exact longitudinal dispersion in this scalar free-space operator. Evanescent frequencies are damped rather than exponentially amplified. This remains a forward, transverse-field model: backward waves and a full vector Maxwell boundary-value solution are outside its scope. Slice discretization and finite transverse grids require convergence checks even when the continuous propagation kernel is exact.

With `propagate=False`, local material interactions remain active but transverse diffraction between slices is omitted. This is a useful approximation for thin objects and a deliberate control in the skyrmion example. It is not equivalent to a full thick-object calculation. Padding, absorbers and region-of-interest options manage finite-grid costs. Compact material stacks reduce memory by storing constant backgrounds and spatial patches. Their numerical equivalence and the adequacy of finite boundaries must be assessed for each geometry.

### 2.3 Polarization and propagation modes

| Selection | Field representation and intended use | Main qualification |
| --- | --- | --- |
| Scalar | One complex field for a selected polarization response | Eigenmode approximation; polarization conversion is not represented |
| Jones | Two coherent transverse complex components | Pure coherent polarization; projected local material response |
| Stokes | Public $[I,Q,U,V]$ fields with coherent Jones carriers | Mixed input polarization is propagated as an incoherent sum of coherent modes |
| Projection control | Material interaction without transverse inter-slice diffraction | Can fail for thick specimens and masks |
| Multislice | Repeated local interaction and angular-spectrum propagation | Forward propagation with finite-grid and slice errors |
| Fraunhofer detector | Exit-wave FFT and detector projection | Far-field assumption; detector curvature is a separate geometry choice |
| Rayleigh–Sommerfeld detector | Finite-distance scalar evaluation at detector coordinates | Computationally expensive; does not make material interaction fully vectorial |

The Stokes interface does not Fourier-transform the four real Stokes components as if they were complex wavefields. It retains the phase-carrying Jones modes needed for spatial interference and adds their intensities incoherently. Pure-state agreement with Jones is therefore expected by construction. The implementation also provides beam-direction projection, optional local-direction estimates, tilted material voxelization, beamstops, coherence/drift approximations, Poisson sampling, readout noise, photon spreading, saturation, defective pixels and cosmic-ray events. These capabilities are broader than the subset validated by the present figures. Detailed interfaces are documented in the repository's [propagation guide](../../docs/light_propagation_modes.md) and [contrast guide](../../docs/optical_contrast_formalisms.md).

## 3. Reproducible demonstrations

### 3.1 Small-period skyrmion tubes: longitudinal interference

We prescribe a triangular lattice of unit-length Néel textures, repeated uniformly through the film thickness. The lattice spacing is 12 nm and the compact texture radius is 3.4 nm. This is a geometrical test specimen, not a micromagnetically equilibrated cobalt structure. At 778 eV, $\lambda=1.593627$ nm and the first reciprocal shell lies at $G=4\pi/(\sqrt{3}a)=0.604600$ rad/nm. The longitudinal mismatch at normal incidence is

$$\Delta q_z=k-\sqrt{k^2-G^2}=0.046633\ \mathrm{rad/nm}.$$

The thickness is chosen as $t=4\pi/\Delta q_z=269.4766$ nm. A uniform finite-thickness tube contributes $t\,\mathrm{sinc}(q_zt/2)$ to the amplitude, with $\mathrm{sinc}(x)=\sin(x)/x$. At normal incidence, the six first-order rod centres therefore lie at the second longitudinal zero. The Ewald sphere misses the strong central lobes, whose first zeros are at $|q_z|=2\pi/t=0.023316$ rad/nm. It does not avoid all scattering: finite illumination broadens peaks, and finite-thickness envelopes have nonzero side lobes.

Figure 1 defines the texture and longitudinal envelope and compares thick-film and 20 nm rocking factors. Figure 2 evaluates a first-Born scalar magnetic-contrast reference directly on physical Ewald surfaces. The active sample rotation is about the laboratory $y$ axis, with angles measured from normal incidence. The $+G_x$ rod passes through its central longitudinal maximum at $\theta_B=\arcsin(G/2k)=4.3974^\circ$. Frames at both signs of this angle and twice this angle demonstrate angle-selective peaks on a shared intensity scale. These are magnetic superlattice diffraction features rather than atomic crystallographic reflections.

The independent Born calculation Fourier-transforms the Gaussian-weighted transverse magnetic contrast, interpolates its complex amplitude, and applies the longitudinal thickness factor. It uses the incident-direction projection of magnetization as a scalar XMCD channel. It does not represent a full polarization-resolved differential cross section.

Figure 4 repeats the qualitative experiment with the production scalar multislice kernel. Both coordinates and magnetization are rotated, with fractional occupancy at planar slab interfaces. The main panels use 192×192 transverse pixels at 0.75 nm pitch and 2 nm longitudinal slices. A weak synthetic medium, $n_0=1$ and $n_c=10^{-6}$, isolates propagation geometry. A second calculation uses unscaled tabulated cobalt channels. The plotted magnetic signal is the squared Fourier amplitude of the difference between textured and uniformly saturated **complex exit fields**. This computational diagnostic is not an experimentally measured helicity-intensity difference.

The weak-medium validation run integrates a neighbourhood of the normal-incidence $+G$ peak. Its multislice signal is approximately 0.00512 of the projection-only control, demonstrating that omitted longitudinal interference changes the prediction substantially. Refining longitudinal slices from 2 to 1 nm changes a selected Bragg-region intensity pattern by approximately $6.83\times10^{-5}$ in relative $L_2$ norm. These values apply to the specified 128×128 validation grid and region, not to an error bound on every image. The corresponding tabulated-Co Bragg-angle transmitted power fraction is approximately 0.229; absorption and refraction must therefore remain part of interpretation. Exact values are saved in `validation.json`.

### 3.2 Assembly in three-dimensional reciprocal space

For a flat detector at distance $L$, the outgoing direction through pixel $(X,Y)$ determines

$$\mathbf q_{lab}=k\left[\frac{(X,Y,L)}{\sqrt{X^2+Y^2+L^2}}-\hat z\right],\qquad
\mathbf q_s=R_y(\theta)^T\mathbf q_{lab}.$$

Figure 3 combines 49 first-Born frames from −12° to +12°. The physical detector uses 193×193 pixels of pitch 13.5 µm at 7 mm distance. Pixel signals are divided by their solid angle before binning. Repeated observations are averaged within voxels and a separate array records coverage. Unmeasured voxels remain undefined, stored as NaN. A separate volume in Figure 4c assembles actual weak-medium multislice signals from a coarser tilt scan including the exact Bragg angles. The code's propagation phase convention requires identifying physical transverse outgoing momenta with negative NumPy FFT frequencies; this sign is handled before rotation.

The resulting volumes locate scattering along the sampled portions of the rods and their thickness envelopes. They do not constitute a three-dimensional real-space reconstruction. Limited angles, one rotation axis, finite detector acceptance and binning leave missing information. Moreover, the magnetic projection factor changes with tilt, so the combined intensity is a record of these measurements rather than automatically one orientation-independent scalar structure factor. In strongly interacting specimens, Born-style interpretation of the assembled intensities requires further caution.

### 3.2.1 Direct three-dimensional Fourier reference

The configurable skyrmion notebooks also construct an explicit voxelized three-dimensional image of the Gaussian-weighted magnetic contrast, with fractional occupancy at the two slab faces. They Fourier-transform each vector component using `scipy.fft.fftn` and multiply amplitudes by the voxel volume. This independent calculation replaces the analytic longitudinal sinc factor with a finite-voxel Fourier transform. The default image grid is 512×256×256 in z,y,x with spacing 2×0.75×0.75 nm; vacuum padding in z resolves the longitudinal Fourier envelope.

Figure 9 compares the complete scalar $|F[m_z-1]|^2$ volume with diffraction assembled from the Born tilt scan, and with the vector FFT sampled at the same Ewald coordinates. For the matched reference, the incident-direction projection acts on the complex Fourier components before squaring, and binning uses the same coverage normalization as the diffraction calculation. The full scalar volume is an orientation-independent reference; it is not expected to coincide with a tilt-dependent XMCD channel at every angle. An optional `mz` setting supplies an orientation-independent scalar control throughout the experiment.

The default matched Born/FFT volume discrepancy is approximately 0.0322 in relative intensity L2 norm, including finite-z-voxel and complex-interpolation error. A padding-refinement test verifies decreasing discrepancy. Empty voxels remain NaN, and common scales, residual slices, three-dimensional views and a longitudinal rod profile make missing coverage visible. A separate comparison uses the actual multislice scan and an explicitly reported single fitted scale. Its current fast-grid shape discrepancy is approximately 0.394, so it is not an independent validation of quantitative Born-limit agreement. Real-space field of view, exit-spectrum interpolation, and the distinction between a lab-fixed illuminating beam and a sample-fixed reference envelope must be converged or matched before attributing this difference to interaction physics.

### 3.3 Normal-incidence Pt/Co Fourier-transform holography

Figure 5 uses an Au(300)/SiN(20)/Pt(2)/Co(10)/Pt(2) stack, with all thicknesses in nm. The Au mask contains an object hole of radius 180 nm; a reference hole of radius 30 nm, displaced by 600 nm, traverses the entire stack. This displacement exceeds the geometrical separation needed to avoid overlap of the ideal reference cross correlation with the central object autocorrelation. A smooth prescribed domain texture supplies the cobalt magnetization.

The 192×192 grid has 10 nm pitch. The notebook exposes the entire material stack and a maximum slice thickness; with the current default of 5 nm, Au, SiN and Co are each subdivided into slices no thicker than 5 nm. Gaussian illumination with amplitude width 650 nm covers both apertures. Matched circular helicities propagate through identical structures. The far field is evaluated on a uniform transverse momentum grid using an orthonormal FFT. We display one helicity's hologram, the intensity difference $I_{CR}-I_{CL}$, and the magnitude of the reference sideband obtained by inverse transforming that difference.

The reconstruction is a correlation with a finite reference aperture; its resolution and amplitude are therefore reference-dependent. Displaying its magnitude discards sign and phase. Neither this display nor a simple helicity subtraction is claimed to recover a calibrated vector magnetization. A physical flat-detector experiment would additionally require geometric resampling before applying this uniform-grid reconstruction.

### 3.4 Interaction modes and energy dependence

Figure 6 compares scalar, Jones and Stokes predictions for the same FTH geometry and incident states. The present normal-incidence material model provides a useful agreement control: the circular basis is an eigenbasis and the linear material channel is zero. The code reports relative $L_2$ differences against Jones in `mode_metrics.json`. Pure-state Stokes matches its Jones carrier by construction, while scalar has a small discrepancy associated with the effective-index approximation. This example does not establish scalar accuracy for general tilted, anisotropic or polarization-converting samples.

Figure 7 sweeps 772–800 eV using eight declared energies in the Co circular-channel window. Real-space sampling remains fixed, hence so does transverse momentum sampling. This corresponds to a common reciprocal-space grid, not unchanged pixels of a stationary detector at all energies. The saved complex reconstruction cube gives a simple spectral-imaging example. Its reference-sideband RMS amplitude depends on the complex optical response, absorption and interference, and should not be interpreted directly as an absorption spectrum. The sparse energy set illustrates the workflow rather than resolving every spectral feature.

### 3.5 Detector artifacts

Figure 8 applies the production detector-effects model to the FTH intensities on their existing reciprocal grid. A common multiplicative scale for both helicities sets a maximum expected count of 20,000. The declared synthetic measurement includes Poisson sampling, two-count readout noise, saturation at 16,000 counts, a central beamstop, hot and cold pixels, and cosmic-ray events. Fixed camera and exposure seeds make the figure repeatable. Readout currently uses a separate legacy NumPy random path, which the experiment helper seeds and restores explicitly.

This example isolates the acquisition stage; it bypasses physical flat-detector projection and uses a stipulated count budget rather than an absolute beamline flux. The noisy sideband demonstrates how missing low-angle data and sensor defects propagate through Fourier reconstruction. No missing-data recovery is performed.

## 4. Validation, limitations, and reproducibility

The supplied checks verify Ewald-sphere membership, orthogonal rotations, the analytical Bragg condition, unit magnetization, correct averaging of repeated reciprocal samples and preservation of unmeasured voxels. Controlled production calculations test longitudinal refinement and compare multislice against the projection approximation. The validation notebook adds lateral refinement from 0.75 to 0.5 nm at fixed field of view; the selected Bragg-region relative intensity difference is approximately $3.11\times10^{-4}$. All figures use declared geometry and shared within-comparison normalizations; raw arrays accompany the rendered figures.

These checks are necessary but not sufficient for an experimental accuracy claim. Further publication work should quantify field-of-view and boundary sensitivity, converge complete rocking curves and FTH reference sampling, measure numerical agreement with the Born limit, and validate tilted Jones calculations against an independent method. An experimental benchmark would be valuable. A phase or intensity agreement between internal code paths sharing the same kernel is a consistency test, not independent validation.

The model omits general backscattering and full electromagnetic interface boundary conditions. Optical constants, assumed magnetic texture and effective-medium approximations can dominate numerical errors. Neither the prescribed skyrmion lattice nor the domain pattern follows from an equilibrium-energy calculation. Detector artifacts are configurable statistical models, not a calibrated response for a named camera. The examples therefore support controlled studies of specified model assumptions, rather than universal predictive accuracy.

## 5. Software and data availability

Source code is available at [github.com/dhuseljic/fomocid](https://github.com/dhuseljic/fomocid), in `src/scattering_calculator`. The manuscript experiments are in `paper/scattering_calculator`; the extended skyrmion tutorial is `tutorials/17_skyrmion_lattice_ewald_rods.ipynb`. A single command regenerates the figure set. PNG/PDF figures, raw NPZ arrays and JSON reports are generated locally. Provenance includes the commit, working-tree status, runtime-library versions, and hashes of simulator sources and optical tables. Results are not silently presented as a clean tagged release when local source edits exist.

Before submission, the authors should archive a clean release with a persistent identifier, finalize software and optical-data licensing, supply authorship and funding statements, and complete the remaining validation. The draft deliberately leaves those unverified metadata and experimental claims open.

## Figure files and captions

1. `fig01_geometry.pdf`: Prescribed Néel tube texture; analytic +G rocking factors for thick and thin films; thick-film longitudinal envelope with the untilted Ewald mismatch marked.
2. `fig02_skyrmion_tilts.pdf`: First-Born scalar magnetic diffraction at five sample tilts, with one shared logarithmic intensity scale.
3. `fig03_reciprocal_space.pdf`: First-Born intensity assembled on rotated Ewald surfaces, a longitudinal maximum projection, and sampling coverage. Empty voxels are not assigned zero intensity.
4. `fig04_multislice_tilts.pdf`, `fig04b_cobalt_tilts.pdf`, `fig04c_multislice_volume.pdf`: Production weak-medium and tabulated-Co scalar multislice comparisons, plus a volume assembled from actual weak-medium multislice signals. Weak and Co panels use separate stated common scales; absolute brightness cannot be compared across those figures.
5. `fig05_fth.pdf`: Pt/Co domain texture and reference location, CR hologram, CR−CL intensity difference, and finite-reference FTH sideband magnitude.
6. `fig06_modes.pdf`: Matched scalar, Jones and pure-state Stokes helicity differences; relative errors refer to Jones on the same grid.
7. `fig07_hyperspectral.pdf`: Co circular optical channels, reconstructed sideband RMS amplitude, and an energy-position section of the complex-reconstruction magnitude on a common grid.
8. `fig08_detector.pdf`: Expected FTH counts, a seeded detector realization, and the resulting noisy FTH sideband.

9. `fig09_fft_volume_comparison.pdf`, `fig09b_fft_volume_3d.pdf`, `fig09c_multislice_fft_comparison.pdf`: full voxelized-object FFT, matched Ewald-sampled reference, coverage, residuals, rod profile, and comparison with actual multislice diffraction.

## Generated figure preview

These links resolve after running the notebooks or experiment script; generated assets are kept outside the source-only version history.

![Skyrmion tilt comparison](results/fig02_skyrmion_tilts.png)

![Ewald-sphere reciprocal-space assembly](results/fig03_reciprocal_space.png)

![Production multislice tilt comparison](results/fig04_multislice_tilts.png)

![Direct 3D FFT comparison](results/fig09_fft_volume_comparison.png)

![Pt/Co Fourier-transform holography](results/fig05_fth.png)

![Co energy-dependent reconstruction](results/fig07_hyperspectral.png)

Bibliographic entries are provided in [references.bib](references.bib).
