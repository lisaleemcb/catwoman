import logging

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as Planck

base_dir = "/Users/emcbride/Datasets/LoReLi"
df = pd.read_pickle(f"{base_dir}/metadata/LoReLi_database_loggedparams.pkl")


def read_cube(path, type=np.float64):
    try:
        cube = np.fromfile(path, dtype=type)
    except FileNotFoundError:
        print(" !!!!!! file not found : " + path)
        cube = np.zeros(256**3)
        print("moving on...")
    shape = np.shape(cube)[0]
    length = int(shape ** (1 / 3)) + 1

    cube = np.reshape(cube, (length, length, length)).T
    shape = np.shape(cube)

    return cube


def read_params(path):
    sim_params = {}

    # Open the text file
    with open(path, "r") as txtfile:
        # Iterate over each line in the text file
        for line in txtfile:
            # Strip leading and trailing whitespace from the line
            line = line.strip()
            # Split the line into key and value using tab as the delimiter
            if "\t" in line:
                value, key = line.split("\t", 1)  # Split only on the first tab
                # Add the key-value pair to the dictionary
                sim_params[key] = value
            # print(f'{key}: {value}')

    return sim_params


def convert_density(field, z):
    f1 = 2.19e9
    f2 = 2.0e30
    f3 = 1.67e-27
    f4 = 3.08e21

    a = 1.0  # Planck.scale_factor(z)

    return field * (f1 * f2) / (f3 * (a * f4) ** 3)


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

    print(
        "No monotonically increasing part of this function. Are you sure this is correct?"
    )
    return np.nan


def unpack_data(spectra_dict):
    data = np.zeros((len(spectra_dict), spectra_dict[0]["P_k"].size))

    # if isinstance(zrange, int):
    #     data = spectra[zrange][key][krange[0]:krange[1]]
    for i in range(len(spectra_dict)):
        data[i] = spectra_dict[i]["P_k"]

    return data


def tension(sim):
    tension = np.zeros_like(sim.k, sim.z)
    for i in range(tension.size):
        t = sim.Pee[i]["P_k"] / sim.Pbb[i]["P_k"]

    return tension


def round_sig_figs(x, sig_figs=5):
    import numbers

    magnitude = np.floor(np.log10(x))  # Compute the order of magnitude for each value
    decimal_places = (
        sig_figs - magnitude.astype(int) - 1
    )  # Determine decimal places for each value

    if isinstance(decimal_places, (int, numbers.Number)):
        return np.round(x, decimal_places)

    if isinstance(decimal_places, (list, np.ndarray)):
        return np.array([np.round(val, dec) for val, dec in zip(x, decimal_places)])


def indexof(arr, target):
    i = np.searchsorted(arr, target)
    if i == 0:
        idx = 0
    elif i == len(arr):
        idx = len(arr) - 1
    else:
        # pick the closer of arr[i-1] or arr[i]
        idx = i if abs(arr[i] - target) < abs(arr[i - 1] - target) else i - 1
    return idx


def setup_logger(logger_name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

    return logger


# from redshifts_fn="metadata/redshift_list.dat",
redshift_keys = {
    "001": 53.48472,
    "002": 40.6952,
    "003": 33.46641,
    "004": 27.8816,
    "005": 24.71079,
    "006": 22.28757,
    "007": 20.36476,
    "008": 18.79519,
    "009": 17.48531,
    "010": 16.37257,
    "011": 15.41341,
    "012": 14.5765,
    "013": 13.83868,
    "014": 13.18242,
    "015": 12.59418,
    "016": 12.06334,
    "017": 11.58143,
    "018": 11.14162,
    "019": 10.73831,
    "020": 10.36687,
    "021": 10.02347,
    "022": 9.704856,
    "023": 9.408288,
    "024": 9.13142,
    "025": 8.872238,
    "026": 8.629,
    "027": 8.400191,
    "028": 8.184488,
    "029": 7.980729,
    "030": 7.787888,
    "031": 7.605059,
    "032": 7.431433,
    "033": 7.266291,
    "034": 7.108987,
    "035": 6.958943,
    "036": 6.815635,
    "037": 6.678593,
    "038": 6.547388,
    "039": 6.421631,
    "040": 6.300971,
    "041": 6.185082,
    "042": 6.07367,
    "043": 5.966464,
    "044": 5.863215,
    "045": 5.763694,
    "046": 5.66769,
    "047": 5.575007,
    "048": 5.485466,
    "049": 5.398898,
    "050": 5.315149,
    "051": 5.234074,
    "052": 5.155539,
    "053": 5.079419,
    "054": 5.005596,
    "055": 4.933961,
    "056": 4.864412,
    "057": 4.796854,
    "058": 4.731195,
    "059": 4.667354,
    "060": 4.605248,
    "061": 4.544805,
    "062": 4.485953,
    "063": 4.428627,
    "064": 4.372763,
    "065": 4.318304,
    "066": 4.265192,
    "067": 4.213376,
    "068": 4.162805,
    "069": 4.113431,
    "070": 4.065211,
    "071": 4.018101,
}
