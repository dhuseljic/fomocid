# Choosing the HDF5 save mode

The independent hologram sweep, hierarchical tree and single-experiment GUI
share a configurable array-compression setting. **LZF is the default.**

| Save mode | Configuration value | Tradeoff |
| --- | --- | --- |
| Fast lossless compression | `"lzf"` | Fast writes and reads, usually larger files than gzip |
| Compact lossless compression | `"gzip"` | Usually smaller files, slower compression; default gzip level 4 |
| Uncompressed | `None` | No compression CPU cost, generally the largest files |

All modes store the same dataset paths, shapes, array values and dtypes. Tests
write and read complete small experiments in all three modes and compare their
exit waves, ideal/detected holograms, support masks and magnetic fields exactly.
Compression does not introduce additional numerical approximation. Independently
of compression, the existing writer stores exit waves as **complex64** and
intensity/optical-magnetization arrays as **float32**; it does not preserve the
full float64/complex128 precision of every intermediate calculation. Small
metadata datasets generally remain uncompressed.

## Independent sweep

At the top of `tutorials/simulate_hologram_sweep.py`, beside the output path:

```python
hdf5_compression = "lzf"  # or "gzip", or None
```

The script transfers this selection into its pipeline configuration before
running. Existing experiment-level output options of other backends are separate.

## Hierarchical tree

Select the mode on the command line:

```bash
PYTHONPATH=src python tutorials/simulate_hologram_tree.py --compression lzf
PYTHONPATH=src python tutorials/simulate_hologram_tree.py --compression gzip
PYTHONPATH=src python tutorials/simulate_hologram_tree.py --compression none
```

The CLI string `none` becomes Python `None`. `--output` still selects the file
path; compression does not change where the file is written.

## Desktop GUI

Expand **Output**, then choose `lzf`, `gzip`, or `None` in **hdf5_compression**.
This affects the next simulation. **Export HDF5** copies the last successful
result without recompressing it; changing the selection afterward does not
rewrite that result.

## Python API and readback

```python
config = HologramPipelineConfig(recipe="Au(100)/SiN(100)/Co(10)",
                               hdf5_compression="lzf")
```

The setting is recorded at `/_pipeline_config/hdf5_compression` (`"none"` when
uncompressed). Inspect the actual dataset filter with h5py:

```python
import h5py
with h5py.File("results.h5", "r") as f:
    dataset = f["00000/CR/exit_wave"]
    print(dataset.compression)  # "lzf", "gzip", or None
    print(dataset.dtype, dataset.shape)
    exit_wave = dataset[()]     # decompression is automatic
```

LZF is provided by standard h5py installations; external HDF5 tools need support
for that filter. Gzip is more broadly supported by general HDF5 tools. Existing
h5py-based project loaders work with all three modes, including older gzip files.

Sweep execution still overwrites its selected output path. Compression is not
an append/resume setting; choose a different output path when keeping an old run.
The separate `simulate_experiment` backend retains its own
`config.outputs.compression` control.
