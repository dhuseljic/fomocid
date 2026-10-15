# Illumination options and coherence

The same `XRayConfig`, `IlluminationConfig` and `SamplePropagatorConfig` control
illumination in notebooks, the experiment lifecycle and the HDF5 runner. Changing
the sample's optical response does not select a different illumination API.

| Choice | Configuration | Meaning |
| --- | --- | --- |
| Energy and flux | `xray.energy`, `photon_flux` | Energy in eV, total sampled incident flux in photons/s |
| Plane wave | `illumination_function=None` or `"plane_wave"` | Constant complex entrance envelope on the sample window |
| Gaussian beam | `illumination_function="gaussian"` | Envelope with configurable focus, position, size and tilt |
| Beam position | `illumination_config["center"]` | `(y,x)` in metres relative to the sample centre |
| Beam width | `illumination_config["fwhm"]` | Intensity FWHM at the Gaussian waist, in metres |
| Focus/defocus | `illumination_config["distance"]` | Gaussian evaluation distance from its waist; retains curvature and Gouy phase |
| Gaussian tilt | `illumination_config["alpha_beam"]` | `(alpha_y,alpha_x)` in radians; changes the spatial phase and beam geometry |
| Pure polarization | `xray.pol` | `CR`, `CL`, `LH`, `LV`, or linear aliases `x`, `y` |
| Linear orientation | `xray.linear_polarization_angle` | Radians from +x towards +y for the linear polarization states |
| General elliptical/partial polarization | Stokes propagation with `input_stokes` | Uniform `[I,Q,U,V]`; use I=1 to preserve configured source flux |
| Simultaneous colors or polarization components | `illumination.spectral_components` | Independently propagated components whose detector intensities add |
| Separate exposures | `illumination.energies_eV`, `polarizations` | Cartesian product of scan settings, not a simultaneous mixture |
| Approximate spatial-coherence loss | `xray.coherence_length` | `(y,x)` lengths in metres, used by the Gaussian acquisition-blur approximation |

The source scales the complex entrance envelope so its sampled squared amplitudes
sum to `photon_flux`. This is independent of array size or beam width; flux lost
outside the modeled sample window is not predicted by that normalization. The
sample pitch and field of view must resolve the beam amplitude and phase. Plane
wave generation does not apply Gaussian focus or tilt parameters.

## Polarization modes

```python
partially_polarized = experiment.with_changes(propagation={
    "propagator_method": "Stokes",
    "propagator_config": {
        **experiment.propagation.propagator_config,
        "input_stokes": (1., 0., 0., .7),
    },
})
```

This state has 70% circular polarization, equivalent to weights .85 and .15 in
CR and CL. A physical Stokes vector satisfies I >= sqrt(Q²+U²+V²). `(1,0,0,0)` is
unpolarized; a vector with equality is pure and can also describe elliptical
polarization. The Stokes coherency matrix is diagonalized into at most two Jones
carriers. Each retains the common entrance wavefront phase during propagation.
Their intensities and Stokes vectors add incoherently. An explicit `input_stokes`
sets polarization instead of `xray.pol`; keep I=1 unless intentionally applying
an additional intensity factor.

## Simultaneous spectral components versus an energy scan

```python
mixed_colors = experiment.with_changes(illumination={
    "spectral_components": (
        sim.SpectralComponentConfig(energy=778., weight=9., pol="CR"),
        sim.SpectralComponentConfig(energy=780., weight=1., pol="CL"),
    ),
})
```

Weights are positive and normalized: the example allocates 90% and 10% of the
total source flux. `energy_factor` can specify a harmonic or relative energy
instead of an absolute energy. A missing `pol` uses the exposure polarization.
Components can share an energy. Their spatial profile settings are shared; each
color evaluates the wavelength-dependent Gaussian field and material response
separately, then propagates onto the same physical detector pixels. An explicit
Stokes input applies its polarization mixture within each color.

The detector reports summed optical intensity while retaining the component
photon-rate images for acquisition. It samples each color independently, applies
its own energy-dependent ADU gain and combines the signals before an optional
nonlinear per-frame response. See the [detector response guide](detector_response.md).
Complex fields at different energies are never added; the model assumes that temporal
interference averages out over the exposure. No ultrafast beating or pulse
waveform is simulated. Per-component fields, spectra and detector coordinates
are retained by the HDF5 runner. By contrast, `energies_eV=(778.,780.)` creates
two exposures, with distinct acquisition draws, rather than one mixed-color
measurement.

## Spatial coherence and coherent superposition

For independent modes a linear optical model gives

```text
I_detector = sum_a w_a |D_a T_a E_a|^2
```

For components coherent at the same frequency, amplitudes must instead be
combined before taking the squared magnitude. The low-level propagators accept
complex `E_in` fields, including supplied coherent superpositions. The standard
illumination configuration exposes Gaussian/plane-wave envelopes and the
polarization/spectral ensembles above; it does not expose arbitrary spatial
coherent-mode maps or a full mutual-intensity propagation solver.

`coherence_length=(np.inf,np.inf)` sets this blur to zero. Finite
`coherence_length` currently controls Gaussian blur during detector acquisition,
not a decomposition of the entrance spatial coherence. It changes acquired
images rather than the stored ideal intensity or exit fields. This approximation
can reduce fringe visibility but is not equivalent to propagating an arbitrary
spatial mode ensemble through the sample, especially for finite-distance
propagation. Photon spreading and readout noise are separate sensor effects.
See the [propagation guide](light_propagation_modes.md) and the
[coherent-mode literature](https://doi.org/10.1107/S1600577522008736).
