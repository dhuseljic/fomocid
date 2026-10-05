# FTH example API

Run these examples in a notebook after the import cell below, or from the
repository root. Sample dimensions in `FTHConfig` are nm; detector lengths are m.
Arrays use row/y then column/x axes, and helicity pairs are ordered **CR, CL**.

```python
from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'src/scattering_calculator').is_dir())
sys.path.insert(0, str(ROOT / 'paper/scattering_calculator'))
import experiments as ex
import workflow as wf
```

## Run the complete chain

```python
sample = ex.FTHConfig(energy_eV=778., max_slice_nm=5., propagate=True)
camera = wf.DetectorGeometry(shape=(128, 128), pixel_size_m=13.5e-6,
                             distance_m=.05, method='fraunhofer')
effects = ex.DetectorEffectsConfig(expected_peak_counts=20_000., noise_seed=17)
case, ideal, measurement = wf.baseline(wf.OUT, sample, camera, effects)
```

The same calculation is available stage by stage:

```python
case = ex.fth_case(config=sample, mode='Jones')
ideal, layout = wf.project(case, camera)
measurement = wf.acquire(ideal, layout, effects)
```

`case['exits']` contains complex sample exit fields `(2, Ny, Nx, 2)` for Jones and
pure-state Stokes, or `(2, Ny, Nx)` for Scalar. `case['images']` contains ideal
FFT intensities on the sample's reciprocal grid. `case['difference']` is CR−CL
on that grid and `case['reconstruction']` is its complex inverse-FFT correlation.
`ideal` is `(2, detector_rows, detector_columns)` on the physical camera.
`measurement` contains `expected`, `measured`, `beamstop`, and the common `scale`.
The expected budget precedes exposure, efficiency and sensor response. It is
not an absolute photon-flux prediction.

## Change sample, beam and material response

| Teaching parameter | Meaning |
| --- | --- |
| `energy_eV` | Photon energy; reloads wavelength and material channels |
| `layers_nm` | Tuple of `(material, thickness_nm)` entries; explicit interfaces |
| `max_slice_nm` | Maximum slice thickness, applied separately to each layer |
| `object_radius_nm` | Object opening through the Au mask |
| `reference_radius_nm`, `reference_xy_nm` | Reference hole through the entire stack |
| `n`, `dx_nm` | Square transverse grid and pitch |
| `beam_sigma_nm` | Gaussian **amplitude** width in exp(−r²/(2σ²)) |
| `domain_period_nm` | Parameter of the prescribed demonstration domains |

Change magnetization or mask depth in `experiments.fth_case` for specimens beyond
this particular builder. General sample interfaces live in `Structure`,
`Apertures3D`, `SampleConfig` and `FrontApertureConfig`; see
[Tutorial 14](../../tutorials/14_end_to_end_scattering_experiment.ipynb).

```python
from scattering_calculator.database.database_loading import material_params
n0, nc, nl = material_params.load_refractive_index('Co', 778.)
```

The bundled Co circular channel is available at 770–805 eV; this example has a
zero linear channel. Generic loading returns zero magnetic channels. Check
availability before using another edge/material for magnetic spectroscopy.

## Select interaction and propagation independently

```python
from dataclasses import replace
scalar = ex.fth_case(config=sample, mode='Scalar')
jones = ex.fth_case(config=sample, mode='Jones')
stokes = ex.fth_case(config=sample, mode='Stokes')  # pure CR/CL carrier control
projection = ex.fth_case(config=replace(sample, propagate=False), mode='Jones')
finite_distance = replace(camera, method='rayleigh_sommerfeld', shape=(24, 24))
rs_ideal, rs_layout = wf.project(jones, finite_distance)
```

`propagate=False` omits transverse diffraction between material slices while
retaining local interaction. `method` changes free-space propagation after the
sample. Direct RS scales with source pixels × detector pixels; start with a
small detector. `footprint_samples=3` enables a 3×3 subpixel average in either
backend. Raw backend normalizations and transverse-coordinate conventions
need matching for quantitative comparisons; per-image peak-normalized displays
only illustrate shapes.

The production configuration names are:

| Operation | Production API selector |
| --- | --- |
| Field representation | `SamplePropagatorConfig(propagator_method="Scalar" / "Jones" / "Stokes", ...)` |
| Multislice | `SamplePropagatorConfig(propagator_config={"propagate": True}, ...)` |
| Mixed input polarization | Stokes with `propagator_config={"input_stokes": [1., 0., 0., .7], ...}` |
| Exit-to-camera propagation | `DetectorConfig(detector_propagation_method="fraunhofer" / "rayleigh_sommerfeld", ...)` |
| Finite pixel footprint | `use_detector_pixel_footprint=True, detector_pixel_footprint_samples=3` |
| Linear small-angle mapping | Fraunhofer with `ignore_flat_detector_curvature=True` |
| Acquisition | `DetectorConfig(measurement_config=..., detector_params=..., artifacts_config=...)` |

These production selectors supplement the teaching wrapper; the wrapper does
not expose mixed input polarization. See the [propagation guide](../../docs/light_propagation_modes.md)
and [contrast guide](../../docs/optical_contrast_formalisms.md) for additional choices.

## Configure acquisition

```python
effects = ex.DetectorEffectsConfig(
    expected_peak_counts=20_000., beamstop_radius_px=3., noise_seed=17,
    measurement=dict(exposure_time=1., number_frames=1, max_counts_per_image=None),
    detector=dict(counts_per_photon=1., quantum_efficiency=1.,
                  readout_noise_average=0., readout_noise_sigma=2.,
                  detector_threshold=16_000.),
    artifacts=dict(sigma_photon=0., camera_seed=31,
                   average_hot_pixels=12., average_cold_pixels=12.,
                   cosmic_rays_per_second=2.),
)
```

The `camera_seed` fixes the persistent defect map; `noise_seed` controls exposure
realizations (the wrapper also seeds and restores legacy readout state).
`sigma_photon=0` disables photon spreading; set a positive width in pixels to
enable it. Zero hot/cold rates and zero cosmic-ray rate disable those artifacts.
Set `beamstop_radius_px=0` to omit the teaching beamstop. Poisson sampling remains
part of acquisition; use the saved ideal intensity when a noise-free prediction
is needed. The two helicities share a single synthetic intensity scale.

## Hyperspectral imaging and validation

```python
scan = replace(sample, energies_eV=tuple(range(772, 801, 2)))
wf.spectral(wf.OUT, scan, camera)
metrics = wf.validate(wf.OUT, sample)
```

The spectral run saves `spectral/hyperspectral.npz` on a common q grid and
`spectral/fixed_detector_series.npz` with physical camera images and per-energy
`qx`, `qy`. Regrid the fixed-camera data before common-q reconstruction; do not
assume unchanged detector pixels represent unchanged q as energy varies.
The current helper's `sideband_rms` key stores root-sum-square amplitude over the
sideband ROI, despite its historical name.

`validate` reports Scalar/Jones/Stokes discrepancies, slice-halving sensitivity,
and the projection-control difference. It asserts pure-state Stokes consistency,
finite nonnegative projected intensity and exact repeatability of seeded counts.
For independent propagation benchmarks, run the production tests listed in
[VALIDATION.md](VALIDATION.md). Validate the observable and geometry you actually
use; this normal-incidence example does not validate every supported option.
