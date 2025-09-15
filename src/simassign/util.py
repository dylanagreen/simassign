# Non-DESI outside imports
import numpy as np

def generate_random_objects(ramin, ramax, decmin, decmax, rng, density=1000):
    area = (ramax - ramin) * np.degrees((np.sin(np.radians(decmax)) - np.sin(np.radians(decmin))))
    n_obj = int(area * density) # 1000 / sq. deg * area

    ra = rng.uniform(ramin, ramax, size=n_obj)
    dec = rng.uniform(decmin, decmax, size=n_obj)

    return ra, dec