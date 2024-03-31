import numpy as np
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
        if np.all(a[:-1] <= a[1:]):
            return i

    print('No monotonically increasing part of this function. Are you sure this is correct?')
    return NaN
