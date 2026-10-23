"""Photon-energy sampling around the bundled cobalt absorption maximum."""
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def co_absorption_peak_energy():
    """Return the tabulated Co L-edge beta maximum energy in eV.

    Use the same Co_beta.txt table as the optical-response loader, rather than
    the circular dichroism maximum or a nominal edge-onset energy.
    """
    path = Path(__file__).resolve().parents[1] / 'database/material_parameter/refractive_indexes/Co/Co_beta.txt'
    data = np.loadtxt(path)
    return float(data[np.argmax(data[:, 1]), 0])


def sample_absorption_edge_energy(rng):
    """Sample N(Co absorption peak, 1 eV), conditioned on peak +/-2 eV.

    Rejection sampling gives a truncated Gaussian without endpoint pileups.
    Pass a NumPy Generator to reproduce an experiment sequence.
    """
    offset = float(rng.normal(0., 1.))
    while abs(offset) > 2.:
        offset = float(rng.normal(0., 1.))
    return co_absorption_peak_energy() + offset
