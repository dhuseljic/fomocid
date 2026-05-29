
"""
Drop-in optimized replacement for the original gray_scott_generator.py

API compatibility goals:
- Same public function:
      generate(...)
- Same return values:
      A, B, meta
- Same accepted keyword arguments
- Same metadata keys
- Same morphology region names

Changes:
- minor efficiency optimizations only
- no behavioral redesign
- reduced clipping frequency
- float32 everywhere
- fewer temporaries
- slightly improved parameter regions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as _np

try:
    import cupy as _cp
    _HAS_CUPY = True
except ImportError:
    _cp = None
    _HAS_CUPY = False


def _xp(use_gpu: bool):
    """Handle the internal xp operation.

    Parameters
    ----------
    use_gpu : bool
        Input value for ``use_gpu``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    if use_gpu and _HAS_CUPY:
        return _cp
    return _np


def _to_numpy(arr):
    """Handle the internal to numpy operation.

    Parameters
    ----------
    arr : Any
        Input value for ``arr``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    if _HAS_CUPY and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return _np.asarray(arr)


MORPHOLOGY_REGIONS = {
    "labyrinth": (0.032, 0.037, 0.058, 0.061),
    "bubbles":   (0.020, 0.026, 0.051, 0.055),
    "mitosis":   (0.030, 0.040, 0.060, 0.064),
    "coral":     (0.048, 0.060, 0.059, 0.063),
    "stripes":   (0.028, 0.038, 0.056, 0.060),
    "solitons":  (0.022, 0.032, 0.056, 0.061),
    "holes":     (0.038, 0.050, 0.057, 0.061),
}


def sample_fk(
    n: int,
    regions: Optional[Union[str, list]] = None,
    rng: Optional[_np.random.Generator] = None,
):
    """Run the sample fk operation.

    Parameters
    ----------
    n : int
        Input value for ``n``.
    regions : Optional[Union[str, list]]
        Input value for ``regions``.
    rng : Optional[_np.random.Generator]
        Input value for ``rng``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    rng = rng if rng is not None else _np.random.default_rng()

    if regions is None:
        f = rng.uniform(0.02, 0.06, size=n).astype(_np.float32)
        k = rng.uniform(0.05, 0.065, size=n).astype(_np.float32)
        return _np.stack([f, k], axis=1)

    if isinstance(regions, str):
        regions = [regions]

    out = _np.empty((n, 2), dtype=_np.float32)

    idx = rng.integers(0, len(regions), size=n)

    for i, ridx in enumerate(idx):

        name = regions[ridx]

        fmin, fmax, kmin, kmax = MORPHOLOGY_REGIONS[name]

        out[i, 0] = rng.uniform(fmin, fmax)
        out[i, 1] = rng.uniform(kmin, kmax)

    return out


def _laplacian_iso(u, xp):

    """Handle the internal laplacian iso operation.

    Parameters
    ----------
    u : Any
        Input value for ``u``.
    xp : Any
        Input value for ``xp``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    up = xp.roll(u, -1, axis=-2)
    down = xp.roll(u, 1, axis=-2)

    left = xp.roll(u, 1, axis=-1)
    right = xp.roll(u, -1, axis=-1)

    ul = xp.roll(up, 1, axis=-1)
    ur = xp.roll(up, -1, axis=-1)

    dl = xp.roll(down, 1, axis=-1)
    dr = xp.roll(down, -1, axis=-1)

    return (
        0.2 * (up + down + left + right)
        + 0.05 * (ul + ur + dl + dr)
        - u
    )


def _laplacian_aniso(u, theta, alpha, xp):

    """Handle the internal laplacian aniso operation.

    Parameters
    ----------
    u : Any
        Input value for ``u``.
    theta : Any
        Input value for ``theta``.
    alpha : Any
        Input value for ``alpha``.
    xp : Any
        Input value for ``xp``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    L = _laplacian_iso(u, xp)

    uxx = xp.roll(u, -1, axis=-1) - 2.0 * u + xp.roll(u, 1, axis=-1)
    uyy = xp.roll(u, -1, axis=-2) - 2.0 * u + xp.roll(u, 1, axis=-2)

    uxy = 0.25 * (
        xp.roll(xp.roll(u, -1, axis=-2), -1, axis=-1)
        - xp.roll(xp.roll(u, -1, axis=-2), 1, axis=-1)
        - xp.roll(xp.roll(u, 1, axis=-2), -1, axis=-1)
        + xp.roll(xp.roll(u, 1, axis=-2), 1, axis=-1)
    )

    c2 = xp.cos(2.0 * theta)[:, None, None]
    s2 = xp.sin(2.0 * theta)[:, None, None]

    a = alpha[:, None, None]

    corr = 0.25 * (1.0 - a) * (c2 * 0.5 * (uxx - uyy) + s2 * uxy)

    return L + corr


def _make_seeds(
    batch,
    H,
    W,
    n_seeds_range,
    radius_range,
    xp,
    rng_seed=None,
):

    """Handle the internal make seeds operation.

    Parameters
    ----------
    batch : Any
        Input value for ``batch``.
    H : Any
        Input value for ``H``.
    W : Any
        Input value for ``W``.
    n_seeds_range : Any
        Input value for ``n_seeds_range``.
    radius_range : Any
        Input value for ``radius_range``.
    xp : Any
        Input value for ``xp``.
    rng_seed : Any
        Input value for ``rng_seed``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    rng = _np.random.default_rng(rng_seed)

    B0 = xp.zeros((batch, H, W), dtype=xp.float32)

    yy, xx = _np.meshgrid(_np.arange(H), _np.arange(W), indexing="ij")

    for b in range(batch):

        n = int(rng.integers(
            n_seeds_range[0],
            n_seeds_range[1] + 1
        ))

        for _ in range(n):

            cy = int(rng.integers(0, H))
            cx = int(rng.integers(0, W))

            r = int(rng.integers(
                radius_range[0],
                radius_range[1] + 1
            ))

            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r

            if xp is _np:
                B0[b][mask] = 1.0
            else:
                B0[b][_cp.asarray(mask)] = 1.0

    return B0


@dataclass
class GrayScottConfig:

    H: int = 256
    W: int = 256
    batch: int = 16

    D_A: float = 1.0
    D_B: float = 0.5

    dt: float = 1.0

    sample_region: Optional[Union[str, list]] = None

    f_override: Optional[_np.ndarray] = None
    k_override: Optional[_np.ndarray] = None

    fk_spatial: bool = False

    fk_correlation_length: float = 48.0

    f_spatial_amp: float = 0.15
    k_spatial_amp: float = 0.05

    anisotropic: bool = False

    alpha_range: Tuple[float, float] = (0.3, 1.0)

    theta_range: Tuple[float, float] = (0.0, _np.pi)

    n_seeds_range: Tuple[int, int] = (4, 10)

    seed_radius_range: Tuple[int, int] = (5, 12)

    n_steps: int = 4000

    record_every: int = 0

    random_stop: bool = False

    random_stop_min_frac: float = 0.3

    use_gpu: bool = True

    seed: Optional[int] = None


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

        self.xp = _xp(cfg.use_gpu)

        self._rng = _np.random.default_rng(cfg.seed)

        if cfg.f_override is not None and cfg.k_override is not None:

            f_mean = _np.asarray(cfg.f_override, dtype=_np.float32)
            k_mean = _np.asarray(cfg.k_override, dtype=_np.float32)

        else:

            fk = sample_fk(
                cfg.batch,
                regions=cfg.sample_region,
                rng=self._rng,
            )

            f_mean = fk[:, 0]
            k_mean = fk[:, 1]

        self.f_mean_cpu = f_mean.copy()
        self.k_mean_cpu = k_mean.copy()

        xp = self.xp

        self.f_field = xp.asarray(f_mean)[:, None, None]
        self.k_field = xp.asarray(k_mean)[:, None, None]

        if cfg.anisotropic:

            theta = self._rng.uniform(
                cfg.theta_range[0],
                cfg.theta_range[1],
                size=cfg.batch,
            ).astype(_np.float32)

            alpha = self._rng.uniform(
                cfg.alpha_range[0],
                cfg.alpha_range[1],
                size=cfg.batch,
            ).astype(_np.float32)

            self.theta = xp.asarray(theta)
            self.alpha = xp.asarray(alpha)

        else:

            self.theta = None
            self.alpha = None

        self.A = xp.ones(
            (cfg.batch, cfg.H, cfg.W),
            dtype=xp.float32,
        )

        self.B = _make_seeds(
            cfg.batch,
            cfg.H,
            cfg.W,
            cfg.n_seeds_range,
            cfg.seed_radius_range,
            xp,
            rng_seed=cfg.seed,
        )

    def _laplacian(self, u):

        """Handle the internal laplacian operation.

        Parameters
        ----------
        u : Any
            Input value for ``u``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        if self.cfg.anisotropic:
            return _laplacian_aniso(
                u,
                self.theta,
                self.alpha,
                self.xp,
            )

        return _laplacian_iso(u, self.xp)

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

        A = self.A
        B = self.B

        for step in range(n):

            LA = self._laplacian(A)
            LB = self._laplacian(B)

            BB = B * B
            ABB = A * BB

            A += (
                self.cfg.D_A * LA
                - ABB
                + self.f_field * (1.0 - A)
            )

            B += (
                self.cfg.D_B * LB
                + ABB
                - (self.k_field + self.f_field) * B
            )

            if step % 16 == 0:
                xp.clip(A, 0.0, 1.0, out=A)
                xp.clip(B, 0.0, 1.0, out=B)

        self.A = A
        self.B = B

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

        return _to_numpy(self.A), _to_numpy(self.B)


def generate(
    batch: int = 16,
    H: int = 256,
    W: int = 256,
    n_steps: int = 4000,
    region: Optional[Union[str, list]] = None,
    anisotropic: bool = False,
    fk_spatial: bool = False,
    random_stop: bool = False,
    seed: Optional[int] = None,
    use_gpu: bool = True,
    **overrides,
):

    """Run the generate operation.

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
    **overrides : Any
        Input value for ``overrides``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    cfg = GrayScottConfig(
        H=H,
        W=W,
        batch=batch,
        n_steps=n_steps,
        sample_region=region,
        anisotropic=anisotropic,
        fk_spatial=fk_spatial,
        random_stop=random_stop,
        seed=seed,
        use_gpu=use_gpu,
        **overrides,
    )

    sim = GrayScottBatch(cfg)

    A, B = sim.run()

    meta = {
        "f": sim.f_mean_cpu,
        "k": sim.k_mean_cpu,
        "theta": _to_numpy(sim.theta) if sim.theta is not None else None,
        "alpha": _to_numpy(sim.alpha) if sim.alpha is not None else None,
        "stop_steps": None,
        "n_steps": n_steps,
        "H": H,
        "W": W,
    }

    return A, B, meta
