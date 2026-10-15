# A modular simulator for coherent diffractive techniques in transmission geometry

**Software-methods manuscript draft.** Authorship, affiliations, funding and an archived release identifier remain to be supplied.

## Abstract

We introduce the scattering calculator in fo-mo-cid as a tool for studying coherent diffractive techniques in transmission geometry. The simulator follows illumination through a structured sample to a detector in four stages: incident-field construction, multislice light–matter interaction, free-space propagation and detector-signal simulation. Scalar refractive-index and Jones dielectric-tensor descriptions provide alternative material-interaction models; a Stokes interface represents polarization mixtures through coherent Jones carriers. Exit-to-detector propagation uses either a Fraunhofer approximation or finite-distance Rayleigh–Sommerfeld integration. A separate acquisition model adds photon statistics and configurable sensor effects. We illustrate the workflow with normal-incidence Fourier-transform holography (FTH), generating the interference of object and reference waves and a synthetic detector image. This example demonstrates the simulator's use and the consequences of its model choices, without restricting the tool to holography.

## 1. A transmission experiment as the organizing example

Coherent transmission measurements connect a specified incident field and sample to a recorded diffraction pattern. The scattering calculator makes this forward calculation explicit and configurable. Its purpose is to help users study how sample structure, optical response, propagation approximations and acquisition conditions affect the measured signal. The contribution is the simulator and its connected workflow; the underlying diffraction and magneto-optical formalisms follow established theory.

Figure 1 introduces the experiment and the four computational stages. A coherent beam illuminates a planar sample at normal incidence along +z. The sample contains an object aperture and a displaced reference aperture in an absorbing mask. Light transmitted through the two regions propagates to a flat detector and interferes to form a hologram. This is the familiar transmission geometry of X-ray FTH [1]. It provides a compact example in which both the sample exit field and the detector interference pattern are physically interpretable.

![Figure 1: normal-incidence transmission geometry and four-stage simulation workflow](results/fth_workflow/fig01_setup_workflow.png)

**Figure 1.** Incident illumination → multislice transmission through the patterned sample → free-space propagation → detector-signal simulation. The specimen front view identifies the object and reference apertures; the side view shows the layer stack and detector distance. Each computational stage exposes its own model choices. Sample geometry, magnetization and optical constants are inputs to the multislice stage.

The baseline specimen is Au(300)/SiN(20)/Pt(2)/Co(10)/Pt(2), with thicknesses in nm. An object aperture of radius 180 nm removes the Au while retaining the underlying film. A reference aperture of radius 30 nm, displaced by (600, 0) nm, passes through the entire stack. A prescribed Co domain texture supplies the magnetization; the simulator does not calculate magnetic equilibrium. The baseline uses 778 eV, a 192 × 192 sample grid with 10 nm pitch, a Gaussian illumination envelope of width 650 nm, slices no thicker than 5 nm and a 256 × 256 detector with 13.5 µm pixels at 5 cm. These are illustrative settings, not a calibrated experimental instrument or established converged discretization.

## 2. Four stages of the forward model

Let **r** = (x,y) denote sample-plane coordinates, λ the vacuum wavelength and k = 2π/λ. The code uses negative spatial propagation phases. In particular, n = 1 − δ − iβ gives attenuation for β > 0. Schematically, the calculation is

$$
\mathbf E_{\mathrm{exit}}=\left[\prod_{j=1}^{N}\mathcal P_j\mathcal T_j\right]\mathbf E_{\mathrm{in}},\qquad
I_{\mathrm{ideal}}=\sum_a w_a|\mathcal D\mathbf E_{\mathrm{exit},a}|^2,\qquad
C=\mathcal A[I_{\mathrm{ideal}}]. \tag{1}
$$

The product is applied in physical slice order. Here T is local material transmission, P is within-sample propagation, D propagates to and samples the detector, and A models acquisition. For vector fields the squared norm sums transverse components. The index a labels mutually incoherent polarization modes with weights w_a; a pure coherent input requires one mode. Keeping these operations separate allows users to change the interaction formalism, propagation approximation or detector response independently.

### 2.1 Stage 1: illumination

The entrance field is specified as a complex amplitude and polarization,

$$
\mathbf E_{\mathrm{in}}(\mathbf r)=A(\mathbf r)e^{i\phi(\mathbf r)}\mathbf p,\qquad
\mathbf p^\dagger\mathbf p=1. \tag{2}
$$

A plane wave uses constant A and phase. At its waist, the Gaussian option has A(r) ∝ exp(−r²/w₀²); away from the waist it includes beam expansion, wavefront curvature and Gouy phase. These are the standard Gaussian-beam and scalar-wave descriptions [3]. Normal incidence uses zero beam tilt. The Gaussian helper normalizes peak amplitude, so its envelope alone does not establish a physical photon flux.

`experimental_conditions/light_beam.py` implements `illumination.plane_wave`, `gauss_beam` and `scalar_to_jones`. Source parameters select energy, beam size, position, waist distance and polarization. The circular states are p_CR = (1,−i)/√2 and p_CL = (1,+i)/√2; horizontal, vertical and arbitrary linear states are also available. Physical flux and exposure are configured for acquisition. The Stokes option additionally accepts a physical input Stokes vector for a uniform polarization mixture; this does not by itself specify arbitrary spatial partial coherence.

### 2.2 Stage 2: multislice transmission through the sample

The sample is divided along z into slices with thickness Δz_j. Material occupancy, aperture depth and magnetization determine the local response in each slice. Energy-dependent material data provide isotropic, circular and linear optical channels (n₀, n_c, n_l). Their separation corresponds to charge and magnetic contributions to resonant optical response [2]. In this Co example, the bundled circular channel is available over 770–805 eV and the linear channel is zero. Generic material loading returns zero magnetic channels; selecting an element does not establish that its dichroic spectrum is available.

**Scalar interaction.** One complex field is transmitted through an effective refractive index:

$$
T_j(\mathbf r)=\exp[-ik n_{\mathrm{eff},j}(\mathbf r)\Delta z_j],\qquad
n_{\mathrm{eff}}=n_0+c_{\mathrm{pol}}m_z n_c+
\ell_{\mathrm{pol}}(m_x^2-m_y^2)n_l. \tag{3}
$$

Here **m** is the real magnetization direction and the polarization coefficients are expectation values of C = [[0,i],[−i,0]] and L = diag(1,−1) in state **p**. Thus c_pol = +1 for CR and −1 for CL, while ℓ_pol = 0 for either circular state. Material occupancy f mixes the index with vacuum as n_voxel = f n_eff + (1−f). The transmission magnitude for an isotropic absorbing slice is exp(−kβΔz). `simple_propagation.scalar_wavefronts.apply_index_slice` implements this interaction.

**Jones interaction.** Two coherent transverse components retain polarization conversion. For normal incidence the implemented projected dielectric tensor is

$$
\boldsymbol\epsilon_\perp=
\begin{pmatrix}
\epsilon_0+\epsilon_l(m_x^2-m_y^2)&i\epsilon_c m_z+2\epsilon_lm_xm_y\\
-i\epsilon_c m_z+2\epsilon_lm_xm_y&\epsilon_0-\epsilon_l(m_x^2-m_y^2)
\end{pmatrix},\quad
\epsilon_0=n_0^2,\quad\epsilon_{c,l}=2n_0n_{c,l}. \tag{4}
$$

The last relations use small-anisotropy dielectric increments. Local transmission is

$$
\mathbf E_j^+=\exp[-ik\Delta z_j\sqrt{\boldsymbol\epsilon_{\perp,j}}]\mathbf E_j^-. \tag{5}
$$

`Jones_propagator.wavefronts.apply_eps_slice` evaluates this matrix function. Occupancy mixing occurs at the dielectric level, with vacuum represented by the identity tensor. Scalar and Jones modes therefore use different material representations; agreement is expected in the appropriate eigenmode and weak-anisotropy limit, rather than guaranteed exactly for arbitrary optical increments. Jones mode is needed when the material mixes polarization components. The code also supports fixed-direction and evolving local-direction tensor projection; these are extensions beyond the normal-incidence demonstration, detailed in the [optical-formalism guide](../../docs/optical_contrast_formalisms.md).

**Stokes representation.** The `Stokes` option exposes S = [I,Q,U,V], where I = |E_x|²+|E_y|², Q = |E_x|²−|E_y|², U = 2 Re(E_x E_y*), and V = 2 Im(E_x E_y*). A deterministic Jones transmission J induces

$$
M_{\mu\nu}=\tfrac12\operatorname{Tr}(\sigma_\mu J\sigma_\nu J^\dagger),\qquad
S'_\mu=\sum_\nu M_{\mu\nu}S_\nu. \tag{6}
$$

The σ matrices are the Hermitian basis for this Stokes convention [4]. `Stokes_propagator` implements the conversions and retains coherent Jones carriers during diffraction. It sums mixed modes incoherently after propagation; Fourier-transforming the four real Stokes images would discard the spatial phase needed for holography.

**Propagation within the sample.** Multislice alternates local transmission with transverse diffraction [5]:

$$
\mathbf E_{j+1}^-=\mathcal F^{-1}\!\left[
H_j(k_x,k_y)\,\mathcal F\{\mathbf E_j^+\}\right],\quad
H_j=\exp[-i\Delta z_j\sqrt{k^2-k_x^2-k_y^2}]. \tag{7}
$$

For evanescent frequencies the branch gives decay. The scalar and Jones `propagate_*_multislice` routines call their respective free-space operators. `propagator_method="Scalar"`, `"Jones"` or `"Stokes"` selects the representation; `propagator_config={"propagate": True}` enables diffraction between slices. Setting `propagate=False` retains local interactions but omits this diffraction, giving a projection-style control. It leaves the exit-to-detector calculation active. Slice thickness, transverse pitch, padding and boundary treatment require convergence checks.

### 2.3 Stage 3: free-space propagation to the detector

The detector calculation acts on the completed exit field. Two exposed backends implement different descriptions of this propagation [3,6]. They do not select the interaction formalism or turn multislice on or off.

**Fraunhofer propagation.** In the far-field approximation, each coherent component has amplitude proportional to the exit-field Fourier transform. For a detector point **R** = (X,Y,L),

$$
I_\Omega(X,Y)\propto\sum_{a,p}w_a
|\widetilde E_{\mathrm{exit},a,p}(q_x,q_y)|^2,\qquad
(q_x,q_y)=k(X,Y)/\sqrt{X^2+Y^2+L^2}. \tag{8}
$$

The intensity is projected onto flat-detector pixels. Their solid angle scales as L dA/(X²+Y²+L²)^(3/2); the code retains the corresponding relative correction. Small-angle mapping instead uses (q_x,q_y) ≈ k(X,Y)/L. Far-field propagation and small-angle detector mapping are separate approximations. A useful far-field criterion is a²/(λL) ≪ 1 for relevant illuminated half-width a; structured-field accuracy must still be checked for the chosen observable.

Select `detector_propagation_method="fraunhofer"`. `detector_hologram.gnomonic_projection` performs physical pixel mapping. `ignore_flat_detector_curvature=True` selects the linear small-angle mapping. Pixel-footprint sampling averages subpixel evaluations, improving integration when fringes vary across a pixel.

**Finite-distance Rayleigh–Sommerfeld propagation.** The alternative evaluates scalar diffraction directly on independently specified detector coordinates:

$$
E(X,Y,L)=\iint E_{\mathrm{exit}}(x,y)\,
\frac{L\,e^{-ikR}(1+ikR)}{2\pi R^3}\,dx\,dy,\quad
R=\sqrt{(X-x)^2+(Y-y)^2+L^2}. \tag{9}
$$

This is the kernel in `detector_propagation.RayleighSommerfeldPropagator.forward`, using zero exterior field outside the supplied source window. Select `detector_propagation_method="rayleigh_sommerfeld"` and retain `ignore_flat_detector_curvature=False`. Each coherent transverse component is propagated before intensities are added. Direct integration retains finite-distance scalar diffraction without the Fraunhofer approximation, at a cost coupling source and detector pixels. It does not impose full vector Maxwell boundary conditions.

The angular-spectrum operator of Eq. (7) describes the same continuous scalar half-space propagation problem with consistent conventions, but the present within-sample FFT routine preserves the sample grid. It is not an exposed independently sampled detector backend. Fresnel propagation is a paraxial approximation discussed in the propagation literature [3,6]; the main detector configuration currently exposes Fraunhofer and Rayleigh–Sommerfeld, rather than a third Fresnel selector.

Coordinate orientation, normalization, source support and pixel integration must be matched before comparing the two backends quantitatively. The current teaching helper documents a transverse reversal between its legacy FFT and RS paths and different raw normalizations; independently rescaled images establish morphology, not absolute-flux agreement.

### 2.4 Stage 4: simulation of the detector signal

The ideal intensity is converted into an acquisition using flux, exposure, efficiency and sensor settings. In a simple limit, for photon-rate density Φ and pixel area A_d,

$$
\mu_d=t\eta_d\int_{A_d}\Phi(X,Y)\,dA,\qquad
N_d\sim\operatorname{Poisson}(\mu_d),\qquad
C_d=\operatorname{clip}(gN_d+b_d+\xi_d,0,C_{\max}),\quad
\xi_d\sim\mathcal N(0,\sigma_r^2). \tag{10}
$$

This separates expected detected photons, photon statistics and digitized counts, following standard camera-noise modeling [7]. Equation (10) describes the elementary limit, not every optional artifact or the complete ordering of the implemented acquisition chain.

The production implementation is `experimental_conditions.detector.detector_hologram.add_noise`, with sensor artifacts in `_add_sensor_artifacts`. The experiment's `detector` configuration controls exposure, frame count, efficiency, counts per photon, readout noise, saturation, photon spreading, persistent hot/cold pixels and cosmic-ray realizations. `beamstop` controls the beamstop shadow. Photon spreading distributes an event over neighboring pixels; persistent sensor defects and exposure-dependent random events play different roles. Recorded seeds permit repeatable synthetic acquisitions.

Gaussian blur can approximate unresolved drift or fringe-visibility loss phenomenologically. It is not equivalent to specifying a general mutual-coherence function at illumination. Likewise, realistic-looking artifacts do not establish quantitative agreement with a particular camera. The production runner uses configured flux and exposure; older paper-figure helpers also offer a separate synthetic display-count budget, which must be identified when used.

## 3. Demonstration: a normal-incidence FTH hologram

Figure 2 follows the same specimen through the four stages: incident intensity, sample geometry and exit field, ideal hologram, and a synthetic detector realization. The optical and detector panels use the same sample and acquisition declaration so that the effect of the signal model is visible.

In the far-field interpretation, write the transmitted field as object O and displaced reference R. Their interference gives

$$
I(\mathbf q)=|\widetilde O|^2+|\widetilde R|^2+
2\operatorname{Re}(\widetilde O\widetilde R^*). \tag{11}
$$

The reference displacement supplies fringes and separates the cross-correlation terms after inverse Fourier transformation [1]. The finite reference aperture limits the recovered spatial detail. The selected 600 nm separation exceeds 3R_object + R_reference = 570 nm, the ideal support-separation condition for this geometry. Residual mask transmission and nonuniform illumination should be assessed in the actual calculation.

Matched CR and CL exposures illustrate magnetic contrast through the circular optical channel. Their difference is a derived hologram observable, while either exposure already demonstrates the complete forward model. An optional FTH reconstruction can check fringe interpretation; it is not required to define the simulator's output or contribution. A sideband magnitude is not a calibrated vector magnetization map.

Figure 3 compares the model choices on this same example: scalar/Jones/pure-state Stokes interaction, multislice enabled/disabled, and Fraunhofer/finite-distance propagation. Each comparison varies one modeling choice with the other inputs fixed. Pure-state Stokes shares its Jones carrier, so agreement is an implementation consistency check. The purely circular response makes this normal-incidence sample a useful scalar control, but cannot establish scalar accuracy for arbitrary anisotropic samples. Propagation comparisons need the coordinate and normalization controls described above. Quantitative differences and convergence results must be supplied from executed, matched calculations rather than inferred from display contrast.

Energy series, common-q reconstruction and additional textures remain useful supplementary examples. They demonstrate reuse of the tool without being required components of the main FTH case study.

## 4. Numerical checks, scope and reproducibility

The model describes forward transmission with transverse fields and local optical response. It omits general backscattering and full electromagnetic interface boundary conditions. Discrete accuracy depends on transverse sampling, source-window extent, slice thickness, detector footprints and optical-data quality. The 30 nm reference radius spans only three baseline sample pixels and particularly needs refinement. Numerical convergence does not validate material spectra, an assumed magnetic texture or an uncalibrated sensor response.

The validation notebook checks nonnegative finite intensities, seeded acquisition repeatability, Jones/pure-state-Stokes consistency and slice refinement. Production tests also compare RS propagation with an analytic circular-aperture result and a padded angular-spectrum calculation, and check its discrete adjoint and detector integration. Executed results and remaining work are recorded in [VALIDATION.md](VALIDATION.md); the manuscript does not yet report a completed convergence study or experimental accuracy claim.

The executable interface is [workflow.py](workflow.py), supported by the sample builder in [experiments.py](experiments.py). The primary notebook defines an `ExperimentConfig`, calls `simulate_experiment(experiment)` and saves the declaration and selected observables to HDF5. Complex exit fields, ideal detector intensity and acquired counts remain distinct data products. [API_GUIDE.md](API_GUIDE.md) supplies runnable calls and units. Run `MPLBACKEND=Agg python paper/scattering_calculator/workflow.py` from the repository root to regenerate the existing figure assets.

The main paper has three figures: (1) sample schematic and four-stage workflow; (2) illumination-to-hologram calculation and synthetic acquisition; (3) controlled comparisons of interaction and propagation choices. Existing reconstruction and hyperspectral figures support supplementary material. The current generator retains its earlier figure numbering; the main-paper selection and final assembly must follow this revised plan. Final publication also requires optical-data provenance and redistribution permissions, author metadata, quantitative convergence results and an archived software release.

## References

1. S. Eisebitt et al., “Lensless imaging of magnetic nanostructures by X-ray spectro-holography,” *Nature* **432**, 885–888 (2004). [doi:10.1038/nature03139](https://doi.org/10.1038/nature03139).
2. J. P. Hannon et al., “X-Ray Resonance Exchange Scattering,” *Physical Review Letters* **61**, 1245–1248 (1988); erratum **62**, 2644 (1989). [doi:10.1103/PhysRevLett.61.1245](https://doi.org/10.1103/PhysRevLett.61.1245). The tensor in Eq. (4) states the software's particular optical-channel parametrization.
3. TU Delft, *Advanced Optical Imaging*, “Wavefield propagation.” [Course derivations](https://qiweb.tudelft.nl/aoi/wavefieldpropagation/wavefieldpropagation/).
4. D. S. Kliger, J. W. Lewis and C. E. Randall, *Polarized Light in Optics and Spectroscopy*, Academic Press (1990). Jones, coherency and Mueller–Stokes representations; V-sign conventions vary between sources.
5. K. Li, M. Wojcik and C. Jacobsen, “Multislice does it all—calculating the performance of nanofocusing X-ray optics,” *Optics Express* **25**, 1831–1846 (2017). [doi:10.1364/OE.25.001831](https://doi.org/10.1364/OE.25.001831).
6. D. M. Paganin, *Coherent X-Ray Optics*, Oxford University Press (2006). Rayleigh–Sommerfeld and approximate propagation; Eq. (9) uses the conjugate spatial-phase convention.
7. European Machine Vision Association, *EMVA Standard 1288, Release 4.0, General* (2021). [Sensor characterization and noise](https://www.emva.org/wp-content/uploads/EMVA1288General_4.0Release.pdf). Cited for noise concepts, not as a claim that the simulator or synthetic camera is EMVA compliant.
