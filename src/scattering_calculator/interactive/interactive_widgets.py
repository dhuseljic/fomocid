import numpy as np
import matplotlib.pyplot as plt
import ipywidgets
from ipywidgets import FloatRangeSlider, FloatSlider, Button, interact, IntSlider


def cimshow(im, **kwargs):
    """Simple 2d image plot with adjustable contrast.

    Returns matplotlib figure and axis created.
    """
    im = np.array(im).astype("float")
    fig, ax = plt.subplots(figsize=(7, 7))
    im0 = im[0] if len(im.shape) == 3 else im
    mm = ax.imshow(im0, **kwargs)

    cmin, cmax, vmin, vmax = np.nanpercentile(im, [0.1, 99.9, 0.001, 99.999])
    # vmin, vmax = np.nanmin(im), np.nanmax(im)
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
