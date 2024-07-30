import numpy as np
import logging
from astropy.cosmology import Planck18 as Planck

def read_cube(path, type=np.float64):
    try :
        cube = np.fromfile(path, dtype=type)
    except FileNotFoundError :
        print(" !!!!!! file not found : "+ path)
        cube = np.zeros(256**3)
        print("moving on...")
    shape = np.shape(cube)[0]
    length = int(shape**(1/3)) +1

    cube =np.reshape(cube, (length,length,length)).T
    shape = np.shape(cube)

    return cube

def convert_density(field, z):
    f1 = 2.19e9
    f2 = 2.0e30
    f3 = 1.67e-27
    f4 = 3.08e21

    a = 1.0 # Planck.scale_factor(z)

    return field * (f1 * f2) / (f3 * (a * f4)**3)

def calc_asymmetry(z_early, z_mid, z_end):
    A = (z_early - z_mid) / (z_mid - z_end)

    return A

def duration(z_early, z_end):

    return z_early - z_end

def find_index(arr):
    for i in range(arr.size - 1):
        a = arr[i:]
       # print(f'array looks like {a}')
        if np.all(a[:-1] < a[1:]):
            return i

    print('No monotonically increasing part of this function. Are you sure this is correct?')
    return NaN

def unpack_data(spectra, key, zrange, krange):
    k0 = krange[0]
    kf = krange[1]
    ksize = kf - k0

    if isinstance(zrange, int):
        data = spectra[zrange][key][krange[0]:krange[1]]
    else:
        z0 = zrange[0]
        zf = zrange[1]
        zsize = zf - z0
        data = np.zeros((zsize, ksize))
        for i, zi in enumerate(range(z0, zf)):
            data[i] = spectra[zi][key][k0:kf]

    return data

def tension(sim):
    tension = np.zeros_like(sim.k, sim.z)
    for i in range(tension.size):
        t = sim.Pee[i]['P_k'] /  sim.Pbb[i]['P_k']

    return tension

def setup_logger(logger_name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

    return logger
