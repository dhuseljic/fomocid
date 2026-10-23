"""Sample physical multilayer recipes shared by independent and tree sweeps."""
import math


def sample_multilayer_recipe(rng):
    """Draw merged AuCr layers, a SiN membrane and a merged CoXY layer.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator controlling reproducible independent parent draws.

    Returns
    -------
    str
        Recipe with thicknesses in nm. Repeat count is discrete uniform from
        7 through 15; total Au/Cr thickness is uniform from 700 to 3000 nm.
        Thicknesses are quantized to 0.1 nm; Au:Cr stays exactly 4:1. SiN is uniform from 50 to 500 nm.
        Co and both added elements are independently uniform from 2 to 30 nm. Ordered
        added elements are drawn without replacement from Pt, Al, Ta, Ir, Cu.
    """
    repeats = int(rng.integers(7, 16))
    total = float(rng.uniform(700., 3000.))
    # Quantize Cr in 0.1 nm units, preserving Au:Cr=4 and the total bounds.
    chromium_ticks = min(max(round(2 * total / repeats),
                             math.ceil(1400 / repeats)), math.floor(6000 / repeats))
    chromium = chromium_ticks / 10
    gold = 4 * chromium
    silicon_nitride = float(rng.uniform(50., 500.))
    cobalt, first_thickness, second_thickness = rng.uniform(2., 30., size=3)
    first, second = rng.choice(('Pt', 'Al', 'Ta', 'Ir', 'Cu'), size=2, replace=False)
    return (f'[Au({gold:.1f})Cr({chromium:.1f})]x{repeats}/'
            f'SiN({silicon_nitride:.1f})/Co({cobalt:.1f})'
            f'{first}({first_thickness:.1f}){second}({second_thickness:.1f})')
