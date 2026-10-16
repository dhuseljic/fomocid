# Material refractive-index datasets

All optical coefficients belong to `material_params`, including the optional
natural and ferroelectric channels. Sample configuration supplies their spatial
order fields and enables the requested effects; it does not supply independent
constant corrections or inline spectra.

The source photon energy selects material data when `SampleConfig.setup()` builds
the sample. A new scan energy or simultaneous spectral component gets its own
material database at its physical energy. Neither detector-coordinate energy nor
an unrelated source polarization setting changes the material dataset selection.

## Location and table format

The default root is
`src/scattering_calculator/database/material_parameter/refractive_indexes`.
`sample.refractive_index_database` can select another root, containing the
background datasets and requested response channels for the materials used.
The same root applies to isotropic, magnetic and nonmagnetic optical response.
`SiN` maps to the `Si3N4` material directory.

Existing isotropic and historical Co XMCD files retain their formats, units and
lookup behavior. The existing three-element `(n0,nc,nl)` interface stays intact.
New named channels live alongside them:

| Dataset filename in `Material/` | Optical channel |
| --- | --- |
| `Material_magnetic_circular.txt` | Circular magnetic increment; overrides the legacy circular value when present |
| `Material_magnetic_linear.txt` | Existing linear magnetic increment |
| `Material_natural_linear.txt` | Structural linear anisotropy |
| `Material_natural_circular.txt` | Reciprocal natural circular response |
| `Material_ferroelectric.txt` | Prescribed polar-axis linear anisotropy |

Each named table has **two header lines**, followed by three numeric columns:

```text
Material response channel; provenance and units
energy_eV  delta  beta
... measured or calculated rows ...
```

The last two columns are **dimensionless**, without the historical Co files'
1e-3 scaling. The loader returns `Delta n(E) = -delta(E) - i beta(E)` using the
existing negative-spatial-phase convention. Signed differential beta values are
allowed. These are response increments, not the full isotropic index.
Energies must be positive and strictly increasing, with at least two finite rows.
Named channels use linear interpolation and reject extrapolation. Historical
base datasets retain their existing interpolation behavior.

No natural or ferroelectric datasets are bundled yet. Merely installing the
extension adds no invented spectra. Missing optional magnetic named files fall
back to the historical magnetic values. Explicitly enabling a natural or
ferroelectric effect requires its dataset; absent or out-of-range data raise a
clear error instead of silently producing zero contrast.

## Spatial fields use the standard sample configuration

```python
response = sim.OpticalResponseConfig(
    material="Co", kind="ferroelectric", axis=polar_axis, strength=order_strength,
)
experiment = experiment.with_changes(sample={
    "optical_responses": (response,),
    "refractive_index_database": "/path/to/material_datasets",
})
```

Here the material name is only an example selector. It requires a real
`Co/Co_ferroelectric.txt` channel at the chosen dataset root; the simulator does
not assume Co has this response. Replace the selector and background data with
the actual specimen's material. Axis and strength maps are independent of
magnetization, as described in the [optical-formalism guide](optical_contrast_formalisms.md).
The empty `optical_responses=()` default preserves the original model.

The HDF5 runner records the sample declarations and, for each loaded optional
channel, its evaluated complex coefficient, physical energy, source path and
SHA-256 hash. Simultaneous colors retain separate channel records. The files
remain the source of the spectra; the records make their use inspectable.
