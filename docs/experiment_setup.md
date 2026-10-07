# Standard scattering-experiment setup

Every maintained scattering example uses the production classes collected in
`scattering_calculator.simulation_pipelines`. There is one experiment definition:
`ExperimentConfig`. No paper-only sample class, notebook-local `Config`, or
separate teaching-camera class is needed.

Declare sections in this order, using the same names in notebooks and scripts:

| Section | Production class | What users set |
| --- | --- | --- |
| `xray` | `XRayConfig` | Energy (eV), photon flux (photons/s), polarization, coherence lengths, optional linear polarization angle |
| `simulation` | `SimulationConfig` | Grid shape `(y,x)` and sample-plane pitch |
| `sample` | `SampleConfig` | Recipe, maximum slice thickness, tilt and specimen metadata |
| `magnetic_pattern` | `MagneticPatternConfig` | Pattern name, physical pattern parameters and magnetic materials |
| `aperture` | `FrontApertureConfig` | Object/reference/slit openings, their locations and depths; `None` retains a film without holes |
| `illumination` | `IlluminationConfig` | Spatial profile, focus, direction, energy list and polarization list |
| `propagation` | `SamplePropagatorConfig` | Jones/Scalar/Stokes and within-sample propagation options |
| `detector` | `DetectorConfig` | Physical camera geometry, exit-to-camera propagation, analyzer, acquisition and sensor response |
| `beamstop` | `BeamstopConfig` | Beamstop shadow geometry, or `None` |
| `outputs` | `OutputConfig` | HDF5 path, observables to save, compression and optional summary panels |

**Units:** physical lengths are metres, angles radians, energy eV, grid sizes
pixels. Recipe thickness literals remain **nm**, for example `"Co(30)"`.
Positions use **(y,x)**; Jones fields have components **(Ex,Ey)**. Linear optical
angles are measured from laboratory +x towards +y. Illumination `center` is a
position in metres relative to the sample centre, not a pixel index.

## Declare an experiment

```python
import numpy as np
from scattering_calculator import simulation_pipelines as sim

experiment = sim.ExperimentConfig(
    xray=sim.XRayConfig(
        energy=778., photon_flux=1., pol="LH",
        linear_polarization_angle=np.deg2rad(30.),
    ),
    simulation=sim.SimulationConfig(
        shape=(256,256), real_space_pixel_size=2e-9,
    ),
    sample=sim.SampleConfig(
        recipe="Co(30)", max_slice_thickness=2e-9, sample_name="Co film",
    ),
    magnetic_pattern=sim.MagneticPatternConfig(
        pattern_type_method="magnon_wave",
        pattern_config={
            "period":40e-9, "amplitude":.1,
            "angle":np.deg2rad(60.), "inplane_angle":0., "phase":0.,
        },
    ),
    aperture=sim.FrontApertureConfig(aperture_method=None),
    illumination=sim.IlluminationConfig(
        illumination_function="gaussian",
        illumination_config={
            "center":(0.,0.), "distance":0., "fwhm":150e-9,
            "alpha_beam":(0.,0.),
        },
        energies_eV=(778.,780.,782.), polarizations=("LH",),
    ),
    propagation=sim.SamplePropagatorConfig(
        propagator_method="Jones",
        propagator_config={"propagate":True},
    ),
    detector=sim.DetectorConfig(
        shape=(128,128), detector_center=(64,64), pixel_size=13.5e-6,
        sample_to_detector_distance=.03,
        detector_propagation_method="fraunhofer",
        analyzer_angle=np.deg2rad(120.),
        measurement_config={"exposure_time":1., "number_frames":1,
                            "max_counts_per_image":None},
        detector_params={"counts_per_photon":1., "quantum_efficiency":1.,
                         "readout_noise_average":0., "readout_noise_sigma":0.,
                         "detector_threshold":1e12},
        artifacts_config={"sigma_photon":0.},
    ),
    beamstop=sim.BeamstopConfig(bs_method=None),
    outputs=sim.OutputConfig(path="outputs/co_magnon.h5", overwrite=True,
        save=("exit_wave","fft","fft_intensity","detector_ideal","detector_measured")),
)
```

`photon_flux=1` gives a convenient unit incident-power reference. The illumination
stage always normalizes **total sampled incident power** to this flux, including
when changing grids. Gaussian `fwhm` describes the intensity width in the current
implementation; its amplitude is `exp(-r²/w²)` with `w=fwhm/sqrt(2 ln 2)`.
Set an actual flux and exposure before interpreting camera counts quantitatively.

The coordinator binds the source and grid to all component stages. Do not repeat
the source, pitch or array shape inside the sample, aperture or illumination.

## Simulate, save, reload and plot

```python
results = sim.simulate_experiment(experiment)
results = sim.load_results(experiment.outputs.path)
figure = sim.plot_results(results, energy_index=0, polarization_index=0)
```

The first command performs setup, sample propagation, detector projection and
configured acquisition, then streams the selected observables into **one HDF5
file**. The detector comes from `experiment.detector`; it is never passed again.
The same runner handles one exposure and scans. `None` scan lists use the single
`xray.energy` and `xray.pol`; explicit illumination lists run their Cartesian
product, with energy first and polarization second. Material optical constants,
wavelength and camera q coordinates refresh for each exposure. Detector-derived
sampling is resolved once at the source reference energy, so an energy scan
keeps the same physical specimen grid. Random generated
specimens use one recorded pattern seed throughout the scan. Acquisitions
use recorded persistent camera/kernel seeds and a per-exposure noise seed and preserve the caller's NumPy RNG state.

Loading and plotting read the file without resimulating. Large arrays load only
when requested:

```python
exit_field = results.read("exit_wave", energy_index=0, polarization_index=0)
camera_series = results.stack("detector_ideal")  # energy, polarization, y, x
qx = results.read("coordinates/qx", energy_index=0, polarization_index=0)
```

To execute physics, saving, reloading and summary plotting in one command:

```python
results, figure = sim.run_experiment(experiment)
```

`outputs.plots` selects the summary panels, and `outputs.figure_path` chooses the
figure filename (by default, next to the HDF5 file). An empty `plots` tuple disables
the combined command's plotting. Existing results require `overwrite=True`;
failed runs leave completed files intact. Explicitly saved metadata-only runs can
use an empty `save` tuple. Save and plot are independent choices.

For component diagnostics, the underlying same engine remains available:

```python
run = sim.ScatteringExperiment(experiment).setup()
wavefront = run.propagate()
ideal_detector = run.detect(noise=False)
```

This lower-level object is useful for intermediate previews and independent
validation; `simulate_experiment` is the standard complete run/save command.

For imported or custom arrays, use the same interface:

```python
run = sim.ScatteringExperiment(experiment).setup(
    magnetization=my_vector_field, mask=my_material_occupancy,
)
wavefront = run.propagate()
```

Magnetization is `(Ny,Nx,3)` or `(Nz,Ny,Nx,3)`; masks are `(Nz,Ny,Nx)` after recipe
subdivision. A supplied 2D vector field explicitly broadcasts through depth.
Generated patterns are assigned to layers matching `magnetic_materials` (Co by
default). The saving runner also accepts `simulate_experiment(experiment,
magnetization=..., mask=...)` and records those external input arrays in HDF5.
Custom depth-dependent textures stay input data, never another configuration class.

For an independent propagation benchmark starting with an existing exit field:

```python
run = sim.ScatteringExperiment.from_exit_wave(experiment, complex_exit_field)
ideal_detector = run.detect()
```

This starts at the exit plane; it does not invent a specimen or apply sample
transmission twice. Source energy and source/detector sampling retain the same
standard sections. Provided finite fields use zero-exterior FFT padding.

## Change settings or scan energy

```python
scan = experiment.with_changes(illumination={
    "energies_eV":(776.,778.,780.), "polarizations":("CR","CL"),
}, outputs={"path":"outputs/co_scan.h5"})
results = sim.simulate_experiment(scan)
```

`with_changes` returns independent component objects. No manual wavelength,
tensor or detector updates are needed. The former top-level `energies_eV` field
is accepted as a compatibility alias; a conflicting illumination energy list
raises an error. New declarations use `illumination.energies_eV`.
Keep reconstruction/ROI/reciprocal-volume binning settings in `analysis`.

## Saved observables and HDF5 layout

Choose `outputs.save` from:

| Observable | Saved quantity |
| --- | --- |
| `exit_wave` | Complex scalar/Jones exit fields; Stokes stores all weighted coherent modes, with modes in a separate trailing axis |
| `exit_wave_for_farfield` | Padded coherent exit plane used for the FFT, including the configured physical background |
| `exit_stokes` | Physical I,Q,U,V exit array, available for Stokes propagation |
| `fft` | Raw complex forward FFT, retaining phase and coherent-mode axes |
| `fft_intensity` | Unfiltered FFT intensity, normalized by Parseval to padded exit power |
| `detector_ideal` | Projected intensity including analyzer, before acquisition, sensor response and beamstop |
| `detector_measured` | Acquired detector counts with exposure, frame averaging, beamstop, noise and artifacts |
| `illumination` | Incident coherent fields (including weighted Stokes modes) |
| `magnetization` / `sample_mask` | Vector texture / material occupancy volume; opt-in because these arrays can be large |
| `reconstruction` | Complex inverse FFT of each unfiltered FFT intensity; polarization differences are downstream analysis |

```text
config_json                 complete declaration and output choices
axes/energies_eV             resolved scan energies
axes/polarizations           resolved polarization states
axes/saved_observables       selected products
provenance/                 source hashes (version/time attributes are on the root)
inputs/                     externally supplied specimen arrays, when provided
runs/000000/                first exposure (energy/polarization index attributes)
  coordinates/              sample/detector positions, camera q, FFT q when saved
  materials/                layer names, thicknesses, energy-specific optical data
  exit_wave                 selected observable datasets, with compression
  detector_ideal
  ...
runs/000001/                next exposure, in energy-major order
```

Physical positions are metres and q coordinates are radians/metre in the file.
Mixed Stokes modes must be combined by summing powers, never amplitudes; the
plotting helper does that automatically. `plot_results` renders saved intensity
panels; volumes and vector textures require an explicit analysis slice.
The standard runner uses configured physical photon flux and acquisition settings;
it does not introduce the older paper helper's synthetic peak-count budget.
Regrid fixed-camera scans before comparing them at common q.

## Component tutorials and batch generation

Focused notebooks expose the same component stages one at a time. Statements
such as `experiment.sample = sample = sim.SampleConfig(...)` register the standard
section while preserving a short alias used by the subsequent diagnostic plots.
They call the same production component `.setup()` methods; a notebook about
mask construction or camera defects can stop before full experiment propagation.

The HDF5 sweep accepts this same `ExperimentConfig`. Its existing paired-CR/CL
backend currently uses a detector-derived grid:

```python
experiment.simulation.other_config = {"grid_mode":"detector", "oversampling":2,
                                      "random_seed":0}
# HologramPipeline(experiment, ranges, output_path, n_samples=...) uses these sections.
```

The backend adapter rejects explicit-grid, arbitrary-linear-angle, analyzer and
slice-subdivision settings that this older paired-CR/CL exporter cannot represent.
Use `ScatteringExperiment` for those experiments. `HologramPipelineConfig` remains
an internal/backward-compatible backend record; maintained notebooks no longer
use its flat constructor to declare experiments.

Historical notebooks in `tutorials/legacy` and the dated paper backup retain their
original formats as immutable archives. All active numbered tutorials and paper
notebooks follow the standard. The shared config/physics tests and notebook-schema
audit guard against reintroducing standalone notebook sample or camera APIs.

## Shared magnetic-pattern controls

Use the same physical name whenever the quantity has the same meaning:

| Parameter | Meaning | Patterns |
| --- | --- | --- |
| `period` | Full spatial repeat, in metres | Stripes, labyrinths (target average full repeat), magnons, smooth domains and Neel lattices |
| `radius` | Core radius, in metres | Random skyrmions, disordered skyrmions and Neel lattices |
| `angle` | Modulation/wavevector direction, radians from +x to +y | Wavy stripes and magnons; stripe walls run perpendicular to this direction |
| `amplitude` | Peak perpendicular component as a fraction of saturation | Magnon modulation |
| `sigma` | Smoothing/transition width, in metres | Existing scalar domain generators |
| `min_separation` | Minimum centre-to-centre distance, in metres | Random circular skyrmions |
| `radius_spread` | Bounded half-range of core-radius variation, in metres | Disordered skyrmions |

A `period=40e-9` stripe pattern has 20 nm single domains: one positive and one
negative domain form the full repeat. A magnon with the same `period` completes
one sinusoidal oscillation in 40 nm. A Neel lattice uses `period` for its nearest
neighbour lattice repeat. Disordered particles use `radius` and density, since
there is no prescribed repeat. Keep separate controls when they describe distinct
quantities, such as the mean magnetization `inplane_angle` and the modulation
`angle`, or a waviness displacement in metres and a dimensionless magnetic
`amplitude`.

Older names remain compatibility aliases. `stripe_width` converts to twice that
value for a stripe/labyrinth `period`; for disordered skyrmions its old diameter
meaning converts to half that value for `radius`. `lattice_spacing` maps to
`period`, `angle_stripes` and `wave_angle` to `angle`, `skyr_radius` to `radius`,
and `oop_amplitude` to `amplitude`. Conflicting aliases raise an error.
The production component translates shared names only at the low-level generator
boundary and converts lengths to pixels once. New notebooks should use shared
names in `MagneticPatternConfig.pattern_config`.

### Parameter scans and stage reuse

A Python **list of alternatives** in a scalar configuration parameter requests a
Cartesian scan. Tuples remain single values. For example:

```python
experiment.sample.recipe = ["Co(20)", "Co(30)"]
experiment.propagation.propagator_method = ["Jones", "Scalar"]
experiment.detector.sample_to_detector_distance = [0.03, 0.06]
experiment.illumination.energies_eV = [778., 790.]
experiment.illumination.polarizations = ["LH", "LV"]
results = sim.simulate_experiment(experiment)
```

This produces 32 exposures. Detector and beamstop alternatives reuse matching
sample-propagation results; sample, illumination, physical energy and propagation
changes require fresh propagation. Cached fields are released after their last
use. Existing structural lists (aperture collections, grid dimensions, positions,
Stokes vectors and sensor length ranges) retain their existing meaning. Output
and analysis selections also remain collections rather than simulation axes.
Source `xray.energy` and `xray.pol` may themselves contain lists; declare each
energy/polarization axis either there or in illumination, not in both places.

`results.parameter_axes` records parameter paths and alternatives. Runs are
ordered by parameter Cartesian product, then energy, then polarization:

```python
wave = results.read("exit_wave", scan_index=1, energy_index=0, polarization_index=0)
cube = results.stack("detector_ideal", scan_index=1)
fig = sim.plot_results(results, scan_index=1)
```

Each HDF5 run records its selections, projection energy and whether sample
propagation was reused. Different recipes or grids can produce differently shaped
arrays; read individual runs rather than stacking incompatible shapes.

### Physical energy and detector coordinate energy

`xray.energy` controls refractive indices, illumination and physical propagation.
`detector.projection_energy` controls the detector's q-coordinate convention.
It defaults to `None`, which follows the physical energy. Set a scalar value to
hold detector q coordinates fixed across an absorption-edge scan, or a list to
compare coordinate conventions without repeating sample propagation:

```python
experiment.detector.projection_energy = 778.  # eV; fixed detector q convention
```

Fraunhofer projection samples the physical exit-field Fourier transform at those
q coordinates. Finite-distance Rayleigh–Sommerfeld diffraction always uses the
physical wavelength; projection energy changes its recorded q coordinates only.
This coordinate override is a comparison convention, not a different physical
photon energy at the detector.

### Simultaneous incoherent spectral components

An energy scan produces separate exposures. A multichromatic beam instead
contains several energies **during the same exposure**:

```python
experiment.xray.energy = 780.
experiment.xray.photon_flux = 1e8  # total photons/s across all components
experiment.illumination.spectral_components = (
    sim.SpectralComponentConfig(energy_factor=1., weight=.95),
    sim.SpectralComponentConfig(energy_factor=2., weight=.05),
)
results = sim.simulate_experiment(experiment)
```

Here the fundamental receives 95% of the incident photon flux and the second
harmonic 5%. Weights must be finite and positive and are normalized by their
sum. `energy_factor` follows the nominal energy at each scan point; use
`energy=1560.` for an absolute energy instead. `pol` can override a component's
polarization; otherwise it follows the exposure's source polarization.
`spectral_components` is a collection of simultaneous colors, never a scan axis.
Omitting it preserves the single-color workflow.

Every color uses its own refractive indices, multislice wavelength and physical
free-space propagation. With `detector.projection_energy=None`, every color also
uses its own detector q projection onto the **same physical pixels**. Ideal
intensities are added after detector projection and any optical analyzer. The
combined exposure then receives one acquisition/sensor/beamstop/artifact pass.
The current scalar quantum efficiency and counts-per-photon settings apply to
all colors equally.

Complex outputs retain a trailing spectral-component axis; flux weights are
already included in the amplitudes. For Jones fields the shape is
`(y, x, polarization, spectral_component)`. Stokes adds its incoherent
polarization-mode axis before the color axis. Scalar fields have shape
`(y, x, spectral_component)`. Sum squared amplitudes, never amplitudes, across
incoherent axes. There is no single coherent exit wave for a multichromatic beam.
`fft_intensity` likewise sums incoherent powers on the common sample Fourier
grid; the detector image is formed by projecting each color individually.

HDF5 `runs/NNNNNN/spectral_components/MMMMMM` groups retain the component energy,
polarization, normalized photon-flux fraction, refractive indices and q
coordinates. If requested, they also retain individual exit fields and ideal
images. The main `detector_ideal` dataset contains their sum. Root run q/material
coordinates refer to the first color; use the component groups for other colors:

```python
fundamental = results.read('spectral_components/000000/detector_ideal')
harmonic = results.read('spectral_components/000001/detector_ideal')
combined = results.read('detector_ideal')
```

See `tutorials/19_multichromatic_illumination.ipynb` for an executable example.
