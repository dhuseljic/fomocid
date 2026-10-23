"""Sample physical multilayer recipes shared by independent and tree sweeps."""


def sample_multilayer_recipe(rng):
    """Draw an Au/Cr multilayer, SiN membrane, Co layer and two distinct caps.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator controlling reproducible independent parent draws.

    Returns
    -------
    str
        Recipe with thicknesses in nm. Repeat count is discrete uniform from
        5 through 30; total Au/Cr thickness is uniform from 700 to 3000 nm.
        Au:Cr thickness is exactly 4:1. SiN is uniform from 50 to 500 nm.
        Co and both caps are independently uniform from 2 to 30 nm. Ordered
        cap elements are drawn without replacement from Pt, Al, Ta, Ir, Cu.
    """
    repeats = int(rng.integers(5, 31))
    total = float(rng.uniform(700., 3000.))
    chromium = total / (5 * repeats)
    gold = 4 * chromium
    silicon_nitride = float(rng.uniform(50., 500.))
    cobalt, first_thickness, second_thickness = rng.uniform(2., 30., size=3)
    first, second = rng.choice(('Pt', 'Al', 'Ta', 'Ir', 'Cu'), size=2, replace=False)
    return (f'[Au({gold:.12g})/Cr({chromium:.12g})]x{repeats}/'
            f'SiN({silicon_nitride:.12g})/Co({cobalt:.12g})/'
            f'{first}({first_thickness:.12g})/{second}({second_thickness:.12g})')
