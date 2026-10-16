# Experimental constrained dipolar magnetic domains

`create_constrained_domain_pattern` is a first experimental scalar relaxation
model for fast synthetic magnetic textures. It generates fully magnetized
+1/-1 output while enforcing the requested sign-area fraction **during every
relaxation step**. It is available in all workflows as
`pattern_type_method="constrained_domain_pattern"`.

It follows the competing-interaction model structure described by
[Jagla, PRE 70, 046204 (2004)](https://arxiv.org/html/cond-mat/0402406): local
saturation, interfacial stiffness and nonlocal dipolar competition evaluated
spectrally. The exact sign-area projection, directional control, clipping and
length calibration are our extensions, not a verbatim paper algorithm.
See the [literature review](magnetic_domain_model_research.md).

## Use

```python
from scattering_calculator.sample_generator import create_constrained_domain_pattern
mz, metadata = create_constrained_domain_pattern(
    (128, 128),
    target_magnetization=0.0,
    characteristic_length=16.0,
    alignment=0.0,
    direction=0.0,
    disorder=0.0,
    initial_state="random",
    initial_order=0.0,
    n_steps=400,
    seed=7,
)
```

Direct lengths are pixels. In `MagneticPatternConfig`,
`characteristic_length` and `disorder_length` are in metres and are converted
at the shared generator boundary. `direction` is a stripe-axis angle in radians,
from +x towards +y in array coordinates.

| Control | Meaning |
|---|---|
| `target_magnetization` | Normalized M/Ms in [-1,1]. The up fraction is (1+M)/2; up-pixel count is rounded to the nearest integer. |
| `characteristic_length` | Target **full repeat/spacing**, not stripe thickness or bubble diameter. Measured using an interpolated radial Fourier-power peak. |
| `alignment`, `direction` | Directional interaction penalty for variations along the stripe axis. At zero alignment the interaction is isotropic. |
| `disorder`, `disorder_length` | Strength and Gaussian correlation scale of quenched random-field pinning. |
| `initial_state`, `initial_order` | Random, stripe or hexagonal preparation and its blend with band-limited random initialization. Preparation is distinct from the directional energy penalty. |
| `wall_stiffness` | Dimensionless strength of the auxiliary field's local double-well potential; default 2. |
| `n_steps`, `time_step` | Maximum relaxation steps per attempt and dimensionless solver time increment. |
| `convergence_tolerance` | Auxiliary-field RMS update criterion; must hold for five consecutive steps. |
| `length_tolerance`, `max_length_iterations` | Relative size tolerance and bounded calibration attempts; defaults 0.15 and 5. One attempt disables length feedback. |

At saturated magnetization the output is uniform, relaxation is skipped and
measured length is `None`. The area target applies to the **full generated
grid**. Cropping with an object-hole aperture can change the observed local
fraction; it does not define a material boundary for this periodic solver.

## Implemented model

For the smooth auxiliary field u, the local energy density is
alpha*(u²-1)²/4. The dimensionless Fourier interaction is

    K(k) = (|k|/q0)² - 2|k|/q0 + alignment*(k·stripe_axis/q0)².

The first two terms represent interfacial stiffness competing with the
long-wavelength dipolar contribution. Their isotropic minimum is at q0.
This uses the long-wave approximation, not a finite-thickness demagnetizing
kernel. The zero mode is fixed separately by the area constraint.

A stabilized semi-implicit spectral update relaxes u, followed by a level-set
shift to match the exact requested sign area and clipping of u to [-1,1].
The final binary sign field is the output magnetization; intermediate amplitude
is an internal wall representation, not reduced magnetization strength.
The projection/clipping extensions do not guarantee monotonic energy descent.
The recorded energy and residual trace is diagnostic, not a proof of convergence
to a physical ground state. Boundary conditions are periodic.

The requested spacing is calibrated by repeated same-seed relaxations with an
adjusted preferred interaction scale. Because topology changes and finite-grid
spectra are discrete, the response need not be monotone. The best measured
attempt is returned. Check **both** `length_target_met` and `converged`; neither
is silently assumed. `calibration_history` records every attempted scale.
The raw single-pixel FFT peak and the radial-peak period are retained separately.

## Measured trial

![Constrained-domain morphology comparison](assets/constrained_domains/morphologies.png)

The reproducible script generates 54 scan trials (three target magnetizations,
three lengths, two alignment strengths, three seeds), plus six named examples.
All are 128×128, with at most 400 steps per calibration attempt. In the 54 scan
trials, 44 met the 15% length tolerance; median runtime was approximately 0.71 s
(range 0.32–1.97 s on this machine). None of these scan trials met the strict
1e-5 residual convergence criterion within that budget. The ordered-stripe
example did converge. These are finite-step textures, not established ground
states. All trials preserved the rounded sign-area constraint. Titles report
the actual achieved length, runtime, size-target flag and convergence flag.

- [Full trial records](assets/constrained_domains/measurements.json), including
  residuals, area counts, energy traces and calibration history.
- [CSV summary](assets/constrained_domains/measurements.csv).
- [Tutorial 22](../tutorials/22_general_magnetic_domains.ipynb) exposes editable
  controls before the procedural baseline examples.

Reproduce with:

```bash
PYTHONPATH=src python scripts/catalogue_constrained_domains.py
```

These are timed scalar-model trials, **not a speed or fidelity comparison to
MuMax3 or another LLG solver**. Material calibration, resolution/time-step
convergence, a finite-thickness kernel and reference micromagnetic comparisons
remain necessary before treating this as a validated surrogate. Vector domain
walls, DMI, chirality, skyrmion topology and physical-time switching dynamics
are outside this first backend.
