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

The default `measurement_config["acquisition_mode"] = "accumulated"` samples
photons for all `number_frames` together and applies charge spreading once.
For N frames, the nonlinear response is evaluated on the estimated average
linear signal, F(S_total/N). Readout and sensor artifacts retain accumulated
frame scaling; ADC clipping is applied to the estimated average output.
This restores the fast accumulated acquisition approach. For nonlinear response
and saturation, it approximates the average rather than reproducing per-frame
statistics: F(mean S_frame) generally differs from mean F(S_frame).

Set `measurement_config["acquisition_mode"] = "per_frame"` to independently
sample, convert, read out and clip each physical frame before averaging. Use
this slower mode when low-count nonlinear bias or saturation statistics matter.
Gain and response callbacks describe one frame in either mode; they do not need
to handle the frame count. With one frame, both modes use the same calculation.

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

## Paired-CR/CL hologram sweep

`tutorials/simulate_hologram_sweep.py` explicitly sets
`detector_params["nonlinear_response"] = None`: the characteristic curve remains
linear by default. Scalar or energy-dependent `counts_per_photon` gives ADU per
detected photon; gain sampling across specimens does not make an individual
camera response nonlinear. Noise, beamstop attenuation, defects and the final
ADC threshold still apply with a linear characteristic curve.

Both the fixed detector configuration and `HologramPipelineRanges.detector_params`
accept the response and gain callbacks directly. They are preserved until
acquisition, rather than invoked as random-parameter factory functions. To
choose a different curve per sample, use a sampler such as `Choice` containing
callbacks or a top-level detector-configuration factory returning a dictionary.

```python
import numpy as np

# Example only: replace with a calibrated response if enabling nonlinearity.
def camera_curve(linear_adu):
    full_scale = 60_000.
    return full_scale * (-np.expm1(-linear_adu / full_scale))

# Edit the fixed dictionary in simulate_hologram_sweep.py before building config:
detector_params["nonlinear_response"] = None  # default: retain linear response
# detector_params["nonlinear_response"] = camera_curve  # optional
measurement_config["acquisition_mode"] = "accumulated"  # default, fast
# measurement_config["acquisition_mode"] = "per_frame"  # explicit frame statistics
```

The sweep carries these settings into the sampled detector/measurement
configuration. `accumulated` approximates the nonlinear response using the
average-frame linear ADU; use `per_frame` when the frame-by-frame nonlinear
statistics are needed. An exposure consists of `number_frames` frames and
`exposure_time` is per frame. Saved detected arrays are averaged across frames.

For specimen `00000`, read:

| Path under `00000/metadata/` | Meaning |
|---|---|
| `detector_params/nonlinear_response` | `"None"` for the default linear response, or the callback's module/qualified name. |
| `detector_params/counts_per_photon` | Constant ADU/photon gain or callback identifier. |
| `detector_response/model` | Explicit `linear` or `nonlinear` transfer flag. |
| `detector_response/CR/energy_eV`, `detector_response/CL/energy_eV` | Physical photon energy used for acquisition. |
| `detector_response/CR/counts_per_photon`, `detector_response/CL/counts_per_photon` | Gain evaluated at that energy, ADU per detected photon. |
| `measurement_config/acquisition_mode` | `accumulated` or `per_frame`. |
| `measurement_config/number_frames`, `measurement_config/exposure_time` | Frame count and seconds per frame. |

The CR/CL fields share one response curve for a specimen. The per-helicity gain
records are numeric even when a callback is used, so analysis does not need to
execute it. Callback names do not contain function source, closure parameters
or calibration tables; share that script/data separately when using a custom
response. Files produced before these additions may lack `detector_response/`.

The sweep and detector defaults use `photon_n_variants=10`. This controls the
number of random single-photon kernel variants per spatial response class;
users can increase it when they need a richer response bank.

Charge spreading preserves individual random photon-response variants while
streaming their counts through one reusable detector image. Their convolution
spectra are summed before a single inverse FFT. This avoids storing a full
image for every variant and preserves the spatial classes, event statistics,
and convolution edge cropping.

With `verbose=True, detailed_logging=True`, each polarization prints detector-noise timings
for expected-photon calculation, photon sampling, charge spreading, nonlinear
response, readout draws, finalization, and total acquisition. These measurements
are also saved under
`00000/metadata/detector_response/{CR,CL}/timing_seconds/`, with keys
`expected_photons`, `photon_sampling`, `charge_spreading`, `nonlinear_response`,
`readout_draw`, `finalization`, and `total`. Replace `00000` with the simulation
index. Finalization includes sensor artifacts, rounding, ADC
clipping, and the optional unmasked output. In `per_frame` mode, component times
are summed over frames; `total` measures the complete acquisition, including
frame assembly. The total includes the component times; do not add it to them.


Optical beamstop attenuation is applied to the expected photon counts **before
Poisson sampling**, charge spreading and nonlinear detector response. Fractional
transmission therefore thins the incident photon process rather than scaling
already-detected ADU. Charge deposited by a transmitted photon may spread into
neighboring shadowed pixels; those pixels are not masked again afterward.
Readout noise, defects and cosmic rays originate in the sensor and remain after
the optical attenuation.

When the optional no-beamstop output is requested, it shares the transmitted
photon events and adds independent Poisson events for the blocked contribution.
Each version receives its own charge/response calculation before shared readout
and artifact realizations. Requesting this extra output does not change the
masked photon draws. The no-beamstop image remains unclipped by the ADC threshold.

The beamstop array is an **opacity mask in [0,1]**: zero is open and one is
fully blocked. Generated masks and finite masks supplied to acquisition are
clipped to these bounds; optical photon transmission is `1 - beamstop`. NaN
and infinite mask values are rejected. This also removes Gaussian-filter
roundoff above one before Poisson sampling.
