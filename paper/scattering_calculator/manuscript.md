# A modular simulator for coherent diffractive techniques in transmission geometry

**Software-methods manuscript draft.** Authorship, affiliations, funding and an archived release identifier remain to be supplied.

## Abstract

We introduce the scattering calculator in fo-mo-cid as a tool for studying coherent diffractive techniques in transmission geometry. The simulator follows illumination through a structured sample to a detector in four stages: incident-field construction, multislice light–matter interaction, free-space propagation and detector-signal simulation. Scalar refractive-index and Jones dielectric-tensor descriptions provide alternative material-interaction models; a Stokes interface represents polarization mixtures through coherent Jones carriers. Optional nonmagnetic terms describe natural linear/circular response and prescribed ferroelectric-axis anisotropy. Exit-to-detector propagation uses either a Fraunhofer approximation or finite-distance Rayleigh–Sommerfeld integration. A separate acquisition model adds photon statistics and configurable sensor effects. We illustrate the workflow with normal-incidence Fourier-transform holography (FTH), generating the interference of object and reference waves and a synthetic detector image. This example demonstrates the simulator's use and the consequences of its model choices, without restricting the tool to holography.

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

**Spatial envelope, wavefront and normalization.** `illumination_function=None` or `"plane_wave"` selects a uniform entrance field; `"gaussian"` selects a Gaussian beam. Its `center` is a displacement in metres in (y,x) order, `fwhm` is the waist intensity width, and `distance` locates the evaluation plane relative to the waist. Defocus retains the complex curvature and Gouy phase. For Gaussian illumination, `alpha_beam=(alpha_y,alpha_x)` specifies tilt in radians and modifies the entrance wavefront and beam geometry. Gaussian focus and tilt settings are not applied by the plane-wave generator. `IlluminationConfig` rescales the sampled entrance power to the configured `xray.photon_flux` in photons/s. Thus the beam helper's peak normalization is intermediate; changing beam width or grid does not silently change this declared flux. The finite window still needs adequate support and sampling.

**Pure and partially polarized illumination.** Named CR, CL, LH and LV states supply pure polarization; `linear_polarization_angle` sets a linear angle from laboratory +x towards +y. The Stokes interface additionally accepts general uniform pure elliptical, partially polarized or unpolarized input through `input_stokes=[I,Q,U,V]`. We use I=1 to preserve the configured source flux. A physical state obeys I ≥ √(Q²+U²+V²), with equality for a pure state. Its polarization coherency matrix is decomposed as

$$
\mathbf C=\sum_{b=1}^{B}v_b\mathbf p_b\mathbf p_b^\dagger,\quad
B\leq2,\quad v_b\geq0,\quad\sum_bv_b=1. \tag{2a}
$$

The simulator propagates √v_b A exp(iφ) **p**_b as coherent Jones carriers and sums their intensities and Stokes vectors afterward [4]. Spatial wavefront phase is retained within every carrier. For example, `[1,0,0,0.7]` corresponds to .85 CR plus .15 CL, while `[1,0,0,0]` is unpolarized. An explicit input Stokes vector supplies the polarization state instead of the named source polarization. These are polarization modes with a shared spatial envelope, not an arbitrary spatial mutual-coherence model.

**Multiple simultaneous components and separate scans.** `illumination.spectral_components` specifies mutually incoherent components by absolute `energy` or relative `energy_factor`, positive `weight`, and optional pure-state `pol`. Weights are normalized and divide the total incident photon flux. Each component independently refreshes wavelength, Gaussian field and optical constants, transmits through the same sample, and reaches the same physical detector pixels. With normalized spectral weights w_a and polarization weights v_ab,

$$
I_{\mathrm{ideal}}(X,Y)=\sum_{a,b}w_av_{ab}
\left\|\mathcal D_a\mathcal T_a\mathbf E_{ab}\right\|^2. \tag{2b}
$$

Here **E**_ab denotes a unit-weight entrance mode and T_a includes multislice propagation. Production carrier amplitudes already contain the square roots of the flux weights, so these weights are not applied twice. Material energy dependence and detector q coordinates belong to each component. Intensities add before one detector acquisition; the model assumes temporal cross terms average out. It does not simulate pulse waveforms or ultrafast beating. Colors may share an energy, and a Stokes mixture can occur within each color. Explicit `input_stokes` applies the same polarization mixture to the components. HDF5 retains separate component products and their detector coordinates.

By contrast, `illumination.energies_eV` and `polarizations` declare separate exposures over their Cartesian product. A two-energy scan produces two measurements, whereas two spectral components produce one summed measurement. An incoherent ensemble follows the mode-intensity sum above [8]. A phase-coherent superposition at one frequency instead requires adding amplitudes before forming intensity; complex `E_in` inputs to the low-level propagators permit such supplied fields. The standard Gaussian/plane-wave configuration does not expose an arbitrary spatial-mode ensemble or a cross-spectral-density solver.

**Spatial partial coherence.** `xray.coherence_length=(ξ_y,ξ_x)` supplies the existing Gaussian acquisition-blur approximation. Schematically,

$$
I_{\mathrm{blur}}=G_{\sigma_y,\sigma_x}*I_{\mathrm{ideal}}. \tag{2c}
$$

The detector implementation maps coherence lengths to blur widths and convolves before photon sampling. This operates during acquisition: it does not alter the stored exit fields or ideal intensity and does not propagate a general mutual-coherence function. Its fringe-visibility approximation must be distinguished from the explicit polarization and spectral mode sums, particularly for finite-distance diffraction. Photon spreading is a separate sensor response. The [illumination guide](../../docs/illumination.md) and first notebook expose these controls and their scope.

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

**Optional nonmagnetic optical response.** Magnetic linear dichroism is already present in Eq. (4). Natural linear/circular dichroism and polar-axis anisotropy are added separately through `sample.optical_responses`, an empty collection by default. Each `OpticalResponseConfig` specifies a target layer material, response kind and spatial axis/strength. Its energy-dependent coefficient is retrieved by `material_params` from the material refractive-index datasets, through the same lifecycle as the magnetic channels; neither constant corrections nor inline spectra are supplied in sample configuration. Matching material layers, including their slices, receive the contribution. This does not modify the bundled magnetic spectra or activate new physics in the baseline FTH example.

For `natural_linear` and `ferroelectric`, the implemented lab-frame response is the uniaxial weak-anisotropy increment

$$
\Delta\boldsymbol\epsilon_{\mathrm{lin}}
=2n_0\Delta n\,s\,\mathbf u\mathbf u^T, \tag{5a}
$$

where **u** is a prescribed unit structural or polar axis, independent of **m**, and s is a dimensionless spatial strength. Projection into the transverse basis retains polarization conversion. Real and imaginary increments represent birefringence and dichroism. Natural linear dichroism can encode structural orientation [10]; the ferroelectric option is a prescribed-axis linear optical model relevant to polar-domain imaging [11]. It does not solve ferroelectric order or switching, and its response is even under **u** → −**u**. Consequently, this term alone cannot distinguish otherwise identical 180-degree domains. The axis magnitude is normalized; polarization-magnitude dependence must be supplied through s. Material-specific background constants and anisotropic spectra remain required inputs.

For `natural_circular`, a reciprocal effective gyration model supplies

$$
G=2n_0\Delta n\,s\,I\quad\text{or}\quad
G=2n_0\Delta n\,s\,\mathbf u\mathbf u^T,\qquad
c_N=\widehat{\mathbf k}^{T}G\widehat{\mathbf k},\qquad
\Delta\epsilon_{\perp,N}=
\begin{pmatrix}0&i c_N\\-i c_N&0\end{pmatrix}. \tag{5b}
$$

The isotropic form uses `axis=None`; a supplied axis selects the uniaxial form, and signed s can reverse handedness. Δn is half the circular index splitting to first order. The lab gyrotropic vector G **k̂** depends on propagation direction, unlike the fixed magnetic vector. This preserves the reciprocal direction-reversal symmetry of the effective response. Physical wavevector-magnitude dependence is included in the supplied energy-dependent coefficients. The isotropic/uniaxial parametrization does not implement every crystal gyrotropy tensor or derive microscopic electric-dipole–quadrupole matrix elements [9].

All contributions are added to the dielectric tensor before applying Eq. (5). Material fractions exclude vacuum, and nominal/local-direction projection uses the same tensor machinery as magnetic propagation. Opting in currently uses dense response channels, with a corresponding memory cost. Named material response tables contain energy, differential δ and differential β alongside the existing material files. Their index increment is −δ−iβ; interpolation rejects extrapolation. Each scan energy and simultaneous color loads its own coefficients. `sample.refractive_index_database` selects the common dataset root for all channels. No natural/ferroelectric tables are bundled yet; enabling an absent channel fails explicitly. Evaluated coefficients, dataset paths and hashes accompany the normal experiment metadata. Scalar mode is available when the specified polarization is an eigenstate of the total response; polarization mixing requires Jones/Stokes. The model-options notebook keeps these controls behind a disabled-by-default switch and requires the corresponding material datasets before activation.

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

The ideal optical image is a photon-rate product. For spectral component a at physical energy E_a and detector pixel d, the expected detected photons in one frame are

$$
\mu_{ad}=t\eta_d\int_{A_d}\Phi_a(X,Y)\,dA,\qquad
N_{ad}\sim\operatorname{Poisson}(\mu_{ad}),\qquad
S_d=\sum_a g(E_a)\,[\mathcal K N_a]_d,\qquad
C_d=\operatorname{clip}\!\left(\operatorname{round}[F(S_d)+b_d+\xi_d],0,C_{\max}\right),
\quad\xi_d\sim\mathcal N(0,\sigma_r^2). \tag{10}
$$

Here g is ADU per detected photon, K is optional photon-event spreading, and F is an optional nonlinear characteristic curve. The default uses a constant g and F(S)=S. This separates photon arrival/detection statistics, charge/signal conversion and electronic noise [7]. Gain does not divide the physical photon mean: without spreading and readout noise, one-energy linear signal has mean gμ and variance g²μ. Quantum efficiency is a separate probability factor, not a second gain.

**Energy-dependent ADU per photon.** `detector.detector_params["counts_per_photon"]` accepts either a positive constant or a callable `gain(energy_eV)`. Energy-dependent CCD counts per detected X-ray photon have been measured, for example in the detector calibration of Fenter et al. [12]. Their coefficient is instrument-specific; direct and scintillator-coupled detectors need appropriate calibration. The simulator evaluates g at every physical source/component energy, including secondary harmonics. A `projection_energy` override changes detector coordinates, not the energy used for material response or gain. The optical ideal image remains independent of this electronic conversion.

For simultaneous colors, each N_a is sampled independently and multiplied by its own g(E_a). Their signals then accumulate on the same pixels. This is a compound-Poisson signal: replacing all photon energies by a single effective gain can change both its mean and variance. Even if two spectra produce the same total photon number, they need not produce the same charge or ADU. This matters in hyperspectral scans and when higher-energy harmonics contaminate a nominal exposure.

**Nonlinear response.** `detector_params["nonlinear_response"]` optionally supplies F as a callable of the per-frame linear ADU array. It returns finite nonnegative ADU on compatible pixel coordinates. This can be a measured lookup, spline or other calibrated characteristic. EMVA 1288 General treats cameras without imposing a linear characteristic [7]; CHEOPSim provides an example of incorporating a measured CCD nonlinear response using a quadratic-spline calibration [13]. Scientific CMOS characterization also requires calibration of camera/readout-mode linearity [14]. No universal CCD/CMOS compression coefficient is assumed.

At one energy, F(gN) can be expressed as a count-dependent effective photon gain. For a mixture, F must act once on S=∑g(E_a)N_a rather than independently on each color. It also acts on each realized frame, not on its photon mean. For M frames, the saved image is

$$
\overline C_d=M^{-1}\sum_{f=1}^{M} C_{df}. \tag{10a}
$$

The same single-frame acquisition code is used for both linear and nonlinear response, including readout and ADC clipping before frame averaging. In general, mean F(S_f) differs from F(mean S_f). The pointwise F model describes a calibrated characteristic curve; it does not by itself simulate blooming, persistence, dead time or the charge-driven neighboring-pixel correlations associated with brighter-fatter effects [15]. Those effects require additional spatial or temporal response models.

The production implementation is `experimental_conditions.detector.detector_hologram.add_noise`, supported by the existing photon-spreading and sensor-artifact functions. The experiment's `detector` configuration controls exposure, frame count, efficiency, conversion, readout noise, thresholds, persistent defects and cosmic-ray realizations. `beamstop` controls the beamstop shadow. Persistent camera response and acquisition draws use recorded seeds. The HDF5 runner saves the evaluated energy/gain pairs and callback identifiers; complete reproducibility also requires retaining the calibration functions and files.

Gaussian acquisition blur remains the spatial-coherence approximation described in Stage 1, separate from sensor charge spreading. `max_counts_per_image` remains an explicit synthetic display budget: it rescales the expected pre-nonlinearity linear ADU using one common photon scaling across colors. It is not a calibrated physical exposure or an inverse nonlinear-response calibration. Physical acquisitions instead use the declared flux, exposure and efficiency. See the [detector-response guide](../../docs/detector_response.md) for runnable configuration examples.

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

8. M. Sanchez del Rio et al., “A fast and lightweight tool for partially coherent beamline simulations in fourth-generation storage rings based on coherent mode decomposition,” *Journal of Synchrotron Radiation* **29**, 1354–1367 (2022). [doi:10.1107/S1600577522008736](https://doi.org/10.1107/S1600577522008736).
9. A. P. Oreshko et al., “X-ray natural circular dichroism in langasite crystal,” *Journal of Synchrotron Radiation* **25**, 222–231 (2018). [doi:10.1107/S1600577517015387](https://doi.org/10.1107/S1600577517015387).
10. A. Apseros et al., “X-ray linear dichroic tomography of crystallographic and topological defects,” *Nature* **636**, 354–360 (2024). [doi:10.1038/s41586-024-08233-y](https://doi.org/10.1038/s41586-024-08233-y).
11. M. Schneider et al., “Linear dichroic soft X-ray microscopy of ferroelectric stripe domains in epitaxial K₀.₆Na₀.₄NbO₃,” preprint (2026). [arXiv:2603.14079](https://arxiv.org/abs/2603.14079). This supports the imaging motivation, not a supplied material calibration.

12. P. Fenter, J. G. Catalano, C. Park and Z. Zhang, “On the use of CCD area detectors for high-resolution specular X-ray reflectivity,” *Journal of Synchrotron Radiation* **13**, 293–303 (2006). [doi:10.1107/S0909049506018000](https://doi.org/10.1107/S0909049506018000).
13. D. Futyan et al., “Expected performances of the Characterising Exoplanet Satellite (CHEOPS). II. The CHEOPS simulator,” *Astronomy & Astrophysics* **635**, A23 (2020). [doi:10.1051/0004-6361/201936616](https://doi.org/10.1051/0004-6361/201936616).
14. “High-precision photometry with a scientific CMOS camera: I lab testing of the Marana camera,” *RAS Techniques and Instruments* (2025). [doi:10.1093/rasti/rzaf049](https://doi.org/10.1093/rasti/rzaf049).
15. P. Antilogus et al., “The brighter-fatter effect and pixel correlations in CCD sensors,” *Journal of Instrumentation* **9**, C03048 (2014). [arXiv:1402.0725](https://arxiv.org/abs/1402.0725).
