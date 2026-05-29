
"""
Fast binary magnetic-domain generator
====================================

Drop-in API-compatible replacement for:

    gray_scott_generator.py

Main goal:
- generate binary magnetic-domain-like patterns VERY quickly

This is NOT Gray-Scott.
It is a spectral Swift–Hohenberg-like model.

API compatibility:
------------------

A, B, meta = generate(...)

Input arguments intentionally mirror the original implementation.

Output:
-------
A : zeros (compatibility placeholder)
B : binary +/-1 domains
meta : dict compatible with original structure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Tuple

import numpy as _np

try:
    import cupy as _cp
    _HAS_CUPY = True
except ImportError:
    _cp = None
    _HAS_CUPY = False


# ============================================================
# utilities
# ============================================================

def _xp(use_gpu=True):

    """Handle the internal xp operation.

    Parameters
    ----------
    use_gpu : Any
        Input value for ``use_gpu``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    if use_gpu and _HAS_CUPY:
        return _cp

    return _np


def _to_numpy(x):

    """Handle the internal to numpy operation.

    Parameters
    ----------
    x : Any
        Input value for ``x``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    if _HAS_CUPY and isinstance(x, _cp.ndarray):
        return _cp.asnumpy(x)

    return _np.asarray(x)


# ============================================================
# config
# ============================================================

@dataclass
class GrayScottConfig:

    H: int = 256
    W: int = 256
    k0: float = 0.085
    eps: float = 0.25
    noise_amp: float = 0.02

    batch: int = 16

    n_steps: int = 300

    sample_region: Optional[Union[str, list]] = None

    use_gpu: bool = True

    seed: Optional[int] = None

    # compatibility-only placeholders

    D_A: float = 1.0
    D_B: float = 0.5

    dt: float = 1.0

    f_override: Optional[_np.ndarray] = None
    k_override: Optional[_np.ndarray] = None

    fk_spatial: bool = False

    fk_correlation_length: float = 48.0

    f_spatial_amp: float = 0.15
    k_spatial_amp: float = 0.05

    anisotropic: bool = False

    alpha_range: Tuple[float, float] = (0.3, 1.0)

    theta_range: Tuple[float, float] = (0.0, _np.pi)

    n_seeds_range: Tuple[int, int] = (3, 12)

    seed_radius_range: Tuple[int, int] = (3, 8)

    record_every: int = 0

    random_stop: bool = False

    random_stop_min_frac: float = 0.3


# ============================================================
# solver
# ============================================================

class GrayScottBatch:

    def __init__(self, cfg):

        """Initialize a GrayScottBatch instance.

        Parameters
        ----------
        cfg : Any
            Input value for ``cfg``.

        Returns
        -------
        None
            The function completes in place.
        """
        self.cfg = cfg
        self.k0 = cfg.k0
        self.eps = cfg.eps
        self.noise_amp = cfg.noise_amp

        self.xp = _xp(cfg.use_gpu)

        xp = self.xp

        rng = _np.random.default_rng(cfg.seed)

        # ====================================================
        # initial field
        # ====================================================

        u0 = rng.standard_normal(
            (cfg.batch, cfg.H, cfg.W)
        ).astype(_np.float32)

        self.u = xp.asarray(u0)

        # ====================================================
        # morphology presets
        # ====================================================

        region = cfg.sample_region

        if isinstance(region, list):
            region = region[0]

        if region in [None, "labyrinth", "stripes"]:

            self.k0 = 0.7
            self.eps = 0.8
            self.noise_amp = 0.02

        elif region in ["bubbles", "solitons"]:

            self.k0 = 1.2
            self.eps = 0.6
            self.noise_amp = 0.01

        else:
            pass
            #self.k0 = 0.085
            #self.eps = 0.25
            #self.noise_amp = 0.02

        # ====================================================
        # Fourier operators
        # ====================================================

        ky = 2.0 * xp.pi * xp.fft.fftfreq(cfg.H)
        kx = 2.0 * xp.pi * xp.fft.fftfreq(cfg.W)

        KX, KY = xp.meshgrid(kx, ky)

        k2 = KX**2 + KY**2

        # shell instability operator

        self.L = (
            self.eps
            - (k2 - self.k0**2)**2
        ).astype(xp.float32)

    # ========================================================
    # evolution
    # ========================================================

    def step(self, n=1):

        """Run the step operation.

        Parameters
        ----------
        n : Any
            Input value for ``n``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        xp = self.xp

        u = self.u

        for _ in range(n):

            U = xp.fft.fft2(u, axes=(-2, -1))

            # linear instability
            U *= xp.exp(self.L[None] * 0.2)

            u = xp.fft.ifft2(
                U,
                axes=(-2, -1)
            ).real

            # nonlinear saturation
            u = u + 0.5 * u - 0.25 * u**3

            # normalize
            std = xp.std(u, axis=(-2, -1), keepdims=True) + 1e-6
            u = u / std

            # weak noise
            noise = self.noise_amp * (
                xp.random.standard_normal(u.shape)
            ).astype(xp.float32)

            u += noise

        self.u = u

    # ========================================================
    # run
    # ========================================================

    def run(self):

        """Run the run operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        self.step(self.cfg.n_steps)

        xp = self.xp

        B = xp.where(self.u >= 0.0, 1.0, -1.0)

        A = xp.zeros_like(B, dtype=xp.float32)

        return _to_numpy(A), _to_numpy(B), self.u


# ============================================================
# public API
# ============================================================

def generate(
    batch: int = 16,
    H: int = 256,
    W: int = 256,
    n_steps: int = 300,
    region: Optional[Union[str, list]] = None,
    anisotropic: bool = False,
    fk_spatial: bool = False,
    random_stop: bool = False,
    seed: Optional[int] = None,
    use_gpu: bool = True,
    k0: float = 0.085,
    eps: float = 0.25,
    noise_amp: float = 0.02,
    **overrides,
):
    """API-compatible binary magnetic-domain generator.

    Returns
    -------
    A : ndarray
        Dummy compatibility field (zeros)

    B : ndarray
        Binary magnetic domains {-1,+1}

    meta : dict
        Metadata dictionary compatible with original API

    Parameters
    ----------
    batch : int
        Input value for ``batch``.
    H : int
        Input value for ``H``.
    W : int
        Input value for ``W``.
    n_steps : int
        Input value for ``n_steps``.
    region : Optional[Union[str, list]]
        Input value for ``region``.
    anisotropic : bool
        Input value for ``anisotropic``.
    fk_spatial : bool
        Input value for ``fk_spatial``.
    random_stop : bool
        Input value for ``random_stop``.
    seed : Optional[int]
        Input value for ``seed``.
    use_gpu : bool
        Input value for ``use_gpu``.
    k0 : float
        Input value for ``k0``.
    eps : float
        Input value for ``eps``.
    noise_amp : float
        Input value for ``noise_amp``.
    **overrides : Any
        Input value for ``overrides``.
    """

    cfg = GrayScottConfig(
        H=H,
        W=W,
        batch=batch,
        n_steps=n_steps,
        sample_region=region,
        use_gpu=use_gpu,
        seed=seed,
        k0=k0,
        eps=eps,
        noise_amp=noise_amp,
        **overrides,
    )

    sim = GrayScottBatch(cfg)

    A, B,u = sim.run()

    meta = {
        "f": None,
        "k": None,
        "theta": None,
        "alpha": None,
        "stop_steps": None,
        "n_steps": n_steps,
        "H": H,
        "W": W,
    }

    return A, B, u,meta


# ============================================================
# self-test
# ============================================================

if __name__ == "__main__":

    A, B, meta = generate(
        batch=4,
        H=256,
        W=256,
        region="labyrinth",
        n_steps=300,
        seed=0,
        use_gpu=True,
        k0=0.7,
        eps=0.8,
        noise_amp=0.02,
    )

    print("A:", A.shape)
    print("B:", B.shape)
    print("unique:", _np.unique(B))
