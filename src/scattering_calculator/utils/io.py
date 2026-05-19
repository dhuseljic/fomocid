"""I/O helpers for saving and loading simulation data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _coerce_object_array(arr: np.ndarray) -> np.ndarray:
    """Cast an object-dtype array to the most specific numeric dtype possible."""
    for dtype in (np.complex128, np.float64):
        try:
            return arr.astype(dtype)
        except (ValueError, TypeError):
            continue
    raise TypeError(
        f"Array with dtype=object could not be cast to a numeric HDF5-compatible type. "
        f"Shape: {arr.shape}, sample element type: {type(arr.flat[0]).__name__}"
    )


def save_simulation_arrays_hdf5(
    file_path: str | Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Save a collection of arrays to an HDF5 file.

    Args:
        file_path: Destination path for the HDF5 file.
        arrays: Mapping of dataset name to array.
        metadata: Optional key-value pairs stored as datasets under a ``metadata/`` group.
        overwrite: If False (default), raise FileExistsError when the file
            already exists.

    Returns:
        Path to the saved file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists() and not overwrite:
        raise FileExistsError(
            f"{file_path} already exists. Pass overwrite=True to replace it."
        )

    skipped = [name for name, array in arrays.items() if array is None]
    if skipped:
        import warnings

        warnings.warn(f"Skipping None arrays: {skipped}", stacklevel=2)

    mode = "w" if overwrite else "x"
    with h5py.File(file_path, mode) as f:
        for name, array in arrays.items():
            if array is None:
                continue
            arr = np.asarray(array)
            if arr.dtype == object:
                arr = _coerce_object_array(arr)
            f.create_dataset(name, data=arr, compression="gzip")

        if metadata:
            for key, value in metadata.items():
                parts = f"metadata/{key}".split("/")
                grp = f
                for part in parts[:-1]:
                    grp = grp.require_group(part)
                if isinstance(value, str):
                    value = np.bytes_(value)
                grp.create_dataset(parts[-1], data=value)

    return file_path
