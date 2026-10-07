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
| `illumination` | `IlluminationConfig` | Spatial envelope, waist, focus, centre and beam direction |
| `propagation` | `SamplePropagatorConfig` | Jones/Scalar/Stokes and within-sample propagation options |
| `detector` | `DetectorConfig` | Physical camera geometry, exit-to-camera propagation, analyzer, acquisition and sensor response |
| `beamstop` | `BeamstopConfig` | Beamstop shadow geometry, or `None` |
| `energies_eV` | tuple | Energy-scan points; each energy rebuilds the optical response |

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
    energies_eV=(778.,780.,782.),
)
```

`photon_flux=1` gives a convenient unit incident-power reference. The illumination
stage always normalizes **total sampled incident power** to this flux, including
when changing grids. Gaussian `fwhm` describes the intensity width in the current
implementation; its amplitude is `exp(-r²/w²)` with `w=fwhm/sqrt(2 ln 2)`.
Set an actual flux and exposure before interpreting camera counts quantitatively.

The coordinator binds the source and grid to all component stages. Do not repeat
the source, pitch or array shape inside the sample, aperture or illumination.

## Execute the same procedure

```python
run = sim.ScatteringExperiment(experiment).setup()
wavefront = run.propagate()
ideal_detector = run.detect(noise=False)
measured_detector = run.detect(noise=True)
```

`setup()` builds optical constants at the source energy, parses/subdivides the
recipe, constructs masks and magnetization, and prepares illumination and detector
coordinates. `propagate()` advances coherent fields through the specimen.
`detect()` applies the selected free-space detector model and, optionally,
acquisition. `run.run(noise=False)` performs the three steps together.

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
default). Custom depth-dependent textures stay analysis/input data, never another
configuration class.

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
crossed = experiment.with_changes(detector={
    "analyzer_angle":experiment.xray.linear_polarization_angle + np.pi/2,
})
for energy in experiment.energies_eV:
    current = experiment.with_changes(xray={"energy":energy})
    run = sim.ScatteringExperiment(current).setup()
    exit_field = run.propagate().exit_wave
    pattern = run.detect()
```

`with_changes` returns independent component objects. Rebuilding at each energy
reloads optical constants and wavelength; it avoids stale tensors and detector q
coordinates. Keep reconstruction/ROI/reciprocal-volume binning settings in
`analysis`, separate from the specimen and instrument sections.

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
