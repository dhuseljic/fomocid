# Single-experiment desktop simulator

From the repository root, with the scientific Python environment activated:

```bash
PYTHONPATH=src python tutorials/simulate_single_experiment_gui.py
```

Equivalent module command:

```bash
PYTHONPATH=src python -m scattering_calculator.gui.single_experiment
```

This is a local Tk desktop application. It needs a graphical desktop session,
Python's `tkinter`, Matplotlib with its Tk backend, NumPy, SciPy, h5py, and the
existing simulator dependencies. The `simulator-gui` package extra declares
Matplotlib, SciPy and h5py (`pip install -e ".[simulator-gui]"`); Tk must be
available in your Python installation. There is no web server. Launch it from the
same environment used for the simulation examples. A headless terminal or SSH
session without display forwarding cannot open the window.

## Setting one experiment

Expand a panel by clicking its heading:

- **Sample structure and grid:** recipe, sample tilt, oversampling, grid limits,
  and optional database path. Recipe thicknesses are in **nm**. The default
  is a fixed merged AuCr/SiN/CoPtAl recipe. Check **Random recipe** to replace
  it with one draw from the shared recipe sampler on each simulation. A fixed
  seed reproduces that draw; change the seed for another recipe.
- **Magnetic pattern:** characteristic length, physical wall sigma, alignment,
  direction, disorder, preparation, iteration and convergence controls, optional
  magnetization target, and classification settings. `target_magnetization=None`
  means free evolution; `-1` or `1` creates saturation. These are explicit single
  experiment inputs: the GUI does not automatically draw an alignment or insert
  occasional saturated parents as the dataset examples do.
- **Holography mask:** OH/RH types, radii, centers, edge widths, taper and other
  aperture settings. Parallel aperture lists must have matching lengths. Expand
  **Aperture preview (depth average)** and click **Refresh aperture preview**
  after edits. It uses the same aperture construction as the simulator, with
  taper, edge smoothing, roughness and hole depths. Brightness is the open
  fraction averaged through physical z, weighted by layer thickness: zero means
  material throughout the depth, one means an opening throughout the depth.
  OH and RH centers are labelled, and axes are in µm. The preview uses at most
  256 pixels per axis, so small holes may be less resolved than in the actual
  simulation. It does not compute a magnetic field or optical transmission.
- **X-rays and illumination:** energy, photon flux, coherence, beam width,
  direction and position. The simulator computes both CR and CL.
- **Detector geometry:** distance, detector size and pitch, beam center,
  projection method and pixel-footprint sampling.
- **Acquisition and sensor response:** exposure time, frame count, noise,
  charge spreading, defects and cosmic rays.
  Linear response and 10 photon-response variants remain the defaults.
- **Beamstop:** beamstop geometry, position and mask settings.
- **Propagation:** Scalar/Jones/Stokes, propagation ROI controls,
  optional diffraction, padding and absorber settings.
- **Optional nonlinear response** (inside acquisition): keep **linear**, or select **saturating** with
  a positive ADU scale. The latter maps per-frame linear ADU `x` to
  `x / (1 + x/scale)` before readout and ADC. Its configuration is saved in the
  detector metadata, with the preset and scale under `metadata/gui/`. The scale is the input at which response is half linear.

Outside recipe strings, lengths are **metres**, angles **radians**, energy
**eV**, and exposure **seconds**. Use `1e-9` for 1 nm, `.05` for 5 cm, and
`1.5708` for approximately 90 degrees. Tuples/lists/dictionaries use Python
literal notation, for example `(256, 256)`, `[(0., 0.), (1e-6, 0.)]`, and
`{'radius': 0.0001}`. `None` is accepted for optional values. Expressions such
as `np.pi` are not executed: enter their numeric value. Dropdowns cover
categorical and boolean settings. Additional dictionary overrides let you set
backend options beyond the displayed defaults; these merge after the individual
entries. Arbitrary Python callbacks and constructed optical-response objects
require the Python API; this editor does not execute code.

The starting example uses a 256×256 detector, 40 mm distance, 778 eV energy,
resolved magnetic walls, a complete three-hole mask and a 3 µm illumination
width. Changes to detector distance, shape or pitch can change the real-space
field. Make sure the chosen field still contains the aperture design; increasing
oversampling improves pixel pitch but does not enlarge the physical field.

## Simulate and inspect

Click **Simulate** to run exactly the selected experiment in a separate process.
The progress log shows the existing backend timings. The interface remains
responsive, and **Cancel** stops the child process. Parameters edited during a
run affect the next run; the running child uses a snapshot.

There is no automatic parameter resampling. If the selected grid exceeds the
allocation limit or another input is invalid, the GUI reports the failure.
Adjust the inputs and run again. It retains the previous successful plots and
exportable result after cancellation or failure.

Four plots show:

1. Wall-smoothed magnetic `mz` multiplied by OH transmission.
2. The exit wave, selectable as amplitude, phase, real or imaginary part.
3. Ideal detector intensity.
4. Detected intensity including the selected acquisition effects.

Choose **CL**, **CR**, **CL−CR** or **CL/CR** to inspect the wave/hologram views. For the
exit-wave difference, amplitude/phase are computed *after* subtracting the
complex fields. For **CL/CR**, complex exit fields are divided before taking
amplitude or phase; ideal/detected views divide their respective intensities.
Zero or non-finite denominators and non-finite ratios are masked rather than
shown as infinities. Very small nonzero denominators can still produce large
ratios. Ratio phase displays `angle(CL/CR)` in radians with an automatic
zero-centered range based on the finite phase values, rather than a full ±π
scale; this reveals small magnetic phase shifts. Ratio amplitude displays
`abs(CL/CR)`. Absolute CR/CL phase retains the ±π scale. Manual color limits
can override the automatic range. Log intensity uses a signed logarithmic scale for negative
differences. The magnetic image is common to both helicities. Axes are pixel
indices for each individual array; the magnetic ROI and full exit wave can have
different shapes. The status area reports physical sample pitch, magnetic state,
bubble/stripe counts, order measurements and the actual recipe. Matplotlib's
toolbar provides rectangle zoom, pan, home/back and figure export. Mouse-wheel
scrolling zooms the result image under the pointer.

The color-control row selects one of the four plots. Choose its colormap and
`linear`, `log`, `symlog` or automatic normalization, optionally enter minimum
and maximum intensity, and click **Apply colors**. Blank limits use the
automatic limits. Each image retains its own settings; color changes preserve
its zoom. Log limits must be positive. Use symlog for signed data such as CL−CR.
**Reset view** restores full image extents without discarding color settings.
The aperture preview also has its own Matplotlib navigation toolbar.

## Export

**Export HDF5** saves the last successful experiment to a selected file. The
complete result stays in a temporary directory until export or application exit;
closing the application removes unexported temporary results. Export before
closing to retain the data. The canonical `00000/` layout contains CR/CL exit
waves, ideal/detected holograms, the smoothed OH field, state measurements,
physical parameters, detector diagnostics and timings. Binary precursors and OH
analysis masks are not saved. See [magnetic analysis](magnetic_pattern_analysis.md)
and [detector response](detector_response.md) for details.

## Relevant controls only

The GUI omits legacy `pattern_config_length` overrides, duplicate top-level
readout/efficiency aliases, the fixed generator selector and Python-only
`optical_responses` objects. Use the canonical controls inside detector settings
and physical lengths inside domain generation settings.

Preparation controls appear only when used: `initial_order` for structured
preparations; worm length, waviness and count for worm preparation with nonzero
initial order; disorder correlation length when disorder is nonzero; and length
tolerance when optional length calibration uses more than one attempt. Hidden
entries do not need validation and their backend defaults remain available.
Old stripe-generator waviness controls are not included. These display changes
do not remove supported options from the Python API.
