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

def read_params(path):
    sim_params = {}

    # Open the text file
    with open(path, 'r') as txtfile:
        # Iterate over each line in the text file
        for line in txtfile:
            # Strip leading and trailing whitespace from the line
            line = line.strip()
            # Split the line into key and value using tab as the delimiter
            if '\t' in line:
                value, key = line.split('\t', 1)  # Split only on the first tab
                # Add the key-value pair to the dictionary
                sim_params[key] = value
               # print(f'{key}: {value}')

    return sim_params

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
    return np.nan

def unpack_data(spectra_dict):
    data = np.zeros((len(spectra_dict), spectra_dict[0]['P_k'].size))

    # if isinstance(zrange, int):
    #     data = spectra[zrange][key][krange[0]:krange[1]]
    for i in range(len(spectra_dict)):
        data[i] = spectra_dict[i]['P_k']

    return data

def tension(sim):
    tension = np.zeros_like(sim.k, sim.z)
    for i in range(tension.size):
        t = sim.Pee[i]['P_k'] /  sim.Pbb[i]['P_k']

    return tension

def round_sig_figs(x, sig_figs=5):
    import numbers
    magnitude = np.floor(np.log10(x))  # Compute the order of magnitude for each value
    decimal_places = sig_figs - magnitude.astype(int) - 1  # Determine decimal places for each value

    if isinstance(decimal_places, (int, numbers.Number)):
        return np.round(x, decimal_places)

    if isinstance(decimal_places, (list, np.ndarray)):
        return np.array([np.round(val, dec) for val, dec in zip(x, decimal_places)])

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
