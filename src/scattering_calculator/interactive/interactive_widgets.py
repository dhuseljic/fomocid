from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import ipywidgets
from ipywidgets import FloatRangeSlider, FloatSlider, Button, interact, IntSlider


def cimshow(
    im: ArrayLike,
    **kwargs,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Display a 2-D image with an interactive contrast slider.

    For 3-D arrays an additional integer slider selects the frame to display.
    All extra keyword arguments are forwarded to ``ax.imshow``.

    Parameters
    ----------
    im : array-like
        2-D image of shape ``(H, W)`` or stack of shape ``(N, H, W)``.
    **kwargs
        Additional keyword arguments passed to :func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    im = np.array(im).astype("float")
    fig, ax = plt.subplots(figsize=(7, 7))
    im0 = im[0] if len(im.shape) == 3 else im
    mm = ax.imshow(im0, **kwargs)

    cmin, cmax, vmin, vmax = np.nanpercentile(im, [0.1, 99.9, 0.001, 99.999])
    sl_contrast = FloatRangeSlider(
        value=(cmin, cmax),
        min=vmin,
        max=vmax,
        step=(vmax - vmin) / 500,
        layout=ipywidgets.Layout(width="500px"),
    )

    @ipywidgets.interact(contrast=sl_contrast)
    def update(contrast):
        mm.set_clim(contrast)

    if len(im.shape) == 3:
        w_image = IntSlider(value=0, min=0, max=im.shape[0] - 1)

        @ipywidgets.interact(nr=w_image)
        def set_image(nr):
            mm.set_data(im[nr])

    return fig, ax
