"""Loaders and path references for material optical-constant databases.

Two file formats are supported:

- **L-edge** (Co): ``#``-prefixed comment headers, space-separated columns
  ``energy_eV  value``.
- **M-edge** (Co, Fe, Ni): tab-separated, European decimal commas, one header
  row with column names
  ``energy  beta_circ  deltabeta  err_dbeta  deltadelta  err_ddelta
  deltabeta_no_corr  deltadelta_no_corr``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Directory references
# ---------------------------------------------------------------------------

_BASE = Path(__file__).parent

COBALT_DIR = _BASE / "Cobalt"
IRON_DIR = _BASE / "Iron"
NICKEL_DIR = _BASE / "Nickel"

# ---------------------------------------------------------------------------
# Known file paths
# ---------------------------------------------------------------------------

# Cobalt L-edge
CO_L_BETA = COBALT_DIR / "Co_L_beta.txt"
CO_L_DELTA = COBALT_DIR / "Co_L_delta.txt"
CO_L_DELTA_BETA = COBALT_DIR / "Co_L_delta_beta.txt"
CO_L_DELTA_DELTA = COBALT_DIR / "Co_L_delta_delta.txt"

# M-edge
CO_M_DATA = COBALT_DIR / "Co_M_data.txt"
FE_M_DATA = IRON_DIR / "Fe_M_data.txt"
NI_M_DATA = NICKEL_DIR / "Ni_M_data.txt"

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_l_edge(filepath: Path | str) -> dict[str, NDArray[np.float64]]:
    """Load a two-column L-edge optical-constant file.

    Lines starting with ``#`` are treated as comments. The remaining rows
    contain whitespace-separated ``energy`` and ``value`` columns.

    Parameters
    ----------
    filepath : Path or str
        Path to the L-edge data file.

    Returns
    -------
    dict with keys:
        ``"energy"`` : ndarray
            Photon energy in eV.
        ``"value"`` : ndarray
            Optical constant value (units given in the file header comment).
    """
    data = np.loadtxt(filepath, comments="#")
    return {"energy": data[:, 0], "value": data[:, 1]}


def load_m_edge(filepath: Path | str) -> dict[str, NDArray[np.float64]]:
    """Load a tabular M-edge optical-constant file.

    The file uses European decimal commas and a single tab-separated header
    row. Columns are: ``energy``, ``beta_circ``, ``deltabeta``, ``err_dbeta``,
    ``deltadelta``, ``err_ddelta``, ``deltabeta_no_corr``,
    ``deltadelta_no_corr``.

    Parameters
    ----------
    filepath : Path or str
        Path to the M-edge data file.

    Returns
    -------
    dict
        One key per column name, each mapping to a float64 ndarray.
    """
    columns = [
        "energy",
        "beta_circ",
        "deltabeta",
        "err_dbeta",
        "deltadelta",
        "err_ddelta",
        "deltabeta_no_corr",
        "deltadelta_no_corr",
    ]
    with open(filepath, "r") as f:
        lines = [
            line.replace(",", ".") for line in f.readlines()[1:]  # skip header
        ]
    data = np.array(
        [[float(v) for v in line.split("\t")] for line in lines if line.strip()]
    )
    return {col: data[:, i] for i, col in enumerate(columns)}


__all__ = [
    "COBALT_DIR",
    "CO_L_BETA",
    "CO_L_DELTA",
    "CO_L_DELTA_BETA",
    "CO_L_DELTA_DELTA",
    "CO_M_DATA",
    "FE_M_DATA",
    "IRON_DIR",
    "NICKEL_DIR",
    "NI_M_DATA",
    "load_l_edge",
    "load_m_edge",
]
