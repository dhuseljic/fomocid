# Photon energy, ADU gain and nonlinear detector response

`detector.detector_params["counts_per_photon"]` accepts either a positive constant
or a callable `gain(energy_eV)`. The result is **ADU per detected photon**, not an
incident flux or quantum efficiency. A scan evaluates the function at each source
energy. Simultaneous colors or harmonics retain their own physical energies and
are sampled independently. A `projection_energy` override changes coordinates,
not the energy used for gain or material response.

```python
# Illustrative proportional calibration, not a universal CCD/CMOS law.
def gain_adu_per_photon(energy_eV):
    return reference_adu_per_photon * energy_eV / reference_energy_eV

experiment = experiment.with_changes(detector={
    "detector_params": {
        **experiment.detector.detector_params,
        "counts_per_photon": gain_adu_per_photon,
    },
})
# A constant remains valid:
# experiment.detector.detector_params["counts_per_photon"] = 10.
```

Replace the function with a measured calibration, an interpolation or a physical
conversion appropriate to the detector. Direct and scintillator-coupled detectors
need different calibrations. Energy-dependent CCD counts per X-ray photon have
been measured explicitly; Fenter et al. report a proportional calibration for
one instrument, not a universal coefficient. [X-ray detector calibration](https://doi.org/10.1107/S0909049506018000).
Quantum efficiency remains a separate probability factor before photon sampling.
Do not include it again in ADU per detected photon.

## Photon statistics and mixed spectra

For pixel d and simultaneous component a, the physical acquisition model is

```text
mu_ad = exposure_s * quantum_efficiency * photon_rate_ad
N_ad ~ Poisson(mu_ad)
S_d = sum_a gain(E_a) * spread(N_a)_d
C_d = F(S_d) + readout_noise_d
```

Here `spread` is the optional existing photon-event spreading model, and F is the
identity by default. The detector thresholds and rounds counts after signal
conversion. The ideal saved images remain optical photon-rate products, before
quantum efficiency, gain and sensor effects. Total incident flux is normalized
at illumination; propagation and detector sampling still need their usual
support/normalization convergence checks.

Gain must **not** divide the physical photon mean before Poisson sampling. At one
energy, without spreading or electronics noise, the analog linear signal has
mean g mu and variance g² mu. The prior physical-flux implementation divided by
g before sampling and multiplied afterward, canceling gain from its mean; this
update corrects that unit conversion. Numerical measurements with nonunit gain
therefore change as intended. Constant gain and linear conversion remain the
standard settings.

`max_counts_per_image` remains an explicit synthetic display budget. It chooses
a common photon scaling such that the pre-nonlinearity expected linear ADU peak
matches that budget. This mode replaces physical flux/exposure normalization;
it is not a calibrated physical exposure or an inversion of a nonlinear curve.
For simultaneous colors the scaling is common to all components, preserving
their relative photon rates.

## Optional nonlinear characteristic curve

Set `detector_params["nonlinear_response"]` to a callable of the accumulated
**per-frame linear ADU array**. It must return finite, nonnegative ADU with a
compatible pixel shape. `None` is the default linear response.

```python
# Illustrative smooth compression; use a calibrated curve for real predictions.
def nonlinear_camera_response(linear_adu):
    return full_scale_adu * (-np.expm1(-linear_adu / full_scale_adu))

experiment = experiment.with_changes(detector={
    "detector_params": {
        **experiment.detector.detector_params,
        "nonlinear_response": nonlinear_camera_response,
    },
})
```

A polynomial, spline or tabulated measured characteristic can be supplied by
the same interface. Keep offsets in the readout/bias settings unless the curve
is intentionally calibrated to include them. Restrict a calibration to its
measured range rather than assuming its polynomial extrapolation is physical.

For a monochromatic exposure, F(gN) can be written as an effective photon gain
F(gN)/N. With mixed energies, accumulated charge/signal matters: photons with
different energies need not contribute equal charge. The simulator therefore
forms S = sum g(E_a)N_a and applies **one** F to S. Applying F separately to each
color or to expected photon means generally gives the wrong result.

If F is nonlinear, each frame is independently sampled, converted, read out and
thresholded before averaging. F(sum S_frame), F(mean S_frame) and
mean F(S_frame) differ. Both linear and nonlinear acquisitions use the same single-frame detector code.
Multiple frames are acquired independently, including their readout noise,
artifacts and ADC clipping, before averaging. This also corrects the earlier
aggregate-clipping approximation for a stack of linear frames.

This is a calibrated characteristic-curve model. It can represent compression
or a measured nonlinear transfer, including a piecewise gain regime supplied in
F. It does not claim to simulate every microscopic sensor effect. Charge-driven
pixel-boundary shifts, blooming, persistence and count-rate dead time need their
own spatial or temporal models; a pointwise gain curve alone does not describe
them. In particular, brighter-fatter effects correlate neighboring pixels and
change spot shapes. [Charge redistribution literature](https://arxiv.org/abs/1402.0725).

## Literature and calibration scope

- EMVA 1288 Release 4.0 General describes camera characteristics without assuming
  a linear response and treats signal-dependent noise propagation. This is the
  general framework behind using a measured curve rather than a universal
  nonlinear law. [EMVA General](https://www.emva.org/wp-content/uploads/EMVA1288General_4.0Release.pdf).
- CHEOPSim models calibrated CCD nonlinearity using the inverse of a measured
  quadratic-spline correction. This is a practical example of a nonlinear
  transfer function in a detector simulator. [Futyan et al. (2020)](https://doi.org/10.1051/0004-6361/201936616).
- Scientific CMOS characterization likewise measures the camera's linearity and
  readout-mode dependence; a CCD-specific calibration should not be transferred
  unchanged to CMOS. [Marana laboratory characterization](https://doi.org/10.1093/rasti/rzaf049).

The HDF5 runner records callback identifiers and the evaluated energy/gain pairs
for each acquired exposure. Optional material channels have their own dataset
hashes. Callback identifiers do not serialize a function's source, closure or
calibration files: retain the defining script and calibration data with the
experiment. Nothing executes automatically when saved results are loaded.
