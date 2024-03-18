from logging import NullHandler
import os
import numpy as np
import pandas as pd

import powerbox as pb
from powerbox import get_power

import catwoman.utils as utils

class Cat:
    def __init__(self,
                sim_n,
                verbose=False,
                load_spectra=False,
                load_params=False,
                load_ion=False,
                load_dens=False,
                path_sim='/Users/emcbride/kSZ/data',
                path_params = 'Pee_spectra_LoReLi/formatted',
                path_Pee = 'LoReLi/ps_ee',
                path_ion = '/Users/emcbride/kSZ/data/LoReLi',
                path_density = '/Users/emcbride/kSZ/data/LoReLi/dens',
                ):

        print(f'Loading sim number {sim_n}...')

        self.sim_n = sim_n
        self.path_sim = path_sim
        self.path_Pee = path_Pee
        self.verbose = verbose

        self.file_nums = self.gen_filenums()
        self.redshifts = self.fetch_redshifts()

        if load_spectra:
            if verbose:
                print("fetching P_ee since you asked so nicely...")
            self.spectra = self.fetch_spectra()

        if load_params:
            if verbose:
                print("fetching params since you asked so nicely...")
            self.path_params = path_params
            self.params = self.fetch_params()
        if load_ion:
            if verbose:
                print("fetching ion cubes since you asked so nicely...")
            self.path_ion = path_ion
            self.ion = self.fetch_ion()
        if load_dens:
            if verbose:
                print("fetching density cubes since you asked so nicely...")
            self.path_density = path_density
            self.density = self.fetch_dens()

        print("Loaded and ready for science!!")

    def gen_filenums(self):
        file_nums = []
        for filename in os.listdir(f'{self.path_sim}/{self.path_Pee}/simu{self.sim_n}/postprocessing/cubes/ps_dtb'):
            basename, extension = os.path.splitext(filename)
            file_nums.append(basename.split('electrons')[1])

        return np.sort(file_nums)

    def fetch_params(self):
        if self.verbose:
            print('Fetching simulations parameters...')
        fn_params = f'runtime_parameters_simulation_{self.sim_n}_reformatted.txt'

        if self.verbose:
            print(f'Now reading in params from {self.path_sim}/{self.path_params}/simu{self.sim_n}/{fn_params}')

        df = pd.read_csv(f'{self.path_sim}/{self.path_params}/simu{self.sim_n}/{fn_params}', sep='\t', header=None)
        params = dict(zip(list(df[1]), list(df[0])))
        params['sim_n'] = self.sim_n

        return params

    def fetch_redshifts(self):
        if self.verbose:
            print('Fetching redshifts...')
        fn_z = f'{self.path_sim}/{self.path_Pee}/simu{self.sim_n}/redshift_list.dat'

        redshifts = {}
        with open(fn_z) as f:
                    for line in f:
                        (val, key) = line.split()
                        redshifts[key] = val

        return redshifts

    def fetch_spectra(self, nbins=512):
        if self.verbose:
            print('Fetching Pee spectra...')
            print(f'Reading in files from {self.path_sim}/{self.path_Pee}/simu{self.sim_n}')

        Pee_list = []
        for n in self.file_nums:
            # if self.verbose:
            #     print(f'Now on file {n}')

            Pee_file = f'{self.path_sim}/{self.path_Pee}/simu{self.sim_n}/postprocessing/cubes/ps_dtb/powerspectrum_electrons{n}.dat'
            P_ee = (0,0)
            if not os.path.isfile(Pee_file):
                raise FileNotFoundError(filename)
            if os.path.isfile(Pee_file):
                P_ee = np.loadtxt(Pee_file).T

                z = 0
                if n in self.redshifts.keys():
                    z = self.redshifts[n]

                spectra_dict = {'file_n': n,
                                'z': float(z),
                                'k': P_ee[0],
                                'P_k': P_ee[1]}
                Pee_list.append(spectra_dict)




        return Pee_list

    def fetch_ion(self, nbins=512):
        if self.verbose:
            print('Fetching ion cube...')

        ion_list = []
        for n in self.file_nums:
            # if self.verbose:
            #     print(f'Now on file {n}')

            ion_file = f'{self.path_ion}/xion/xion_256_out{n}.dat'

            if self.verbose:
                print(f'Now reading in ion box from from {ion_file}')

            if not os.path.isfile(ion_file):
                raise FileNotFoundError(ion_file)
            if os.path.isfile(ion_file):
                ion = utils.read_cube(ion_file)

            z = 0
            if n in self.redshifts.keys():
                z = self.redshifts[n]

            ion_dict = {'file_n': n,
                        'z': float(z),
                        'cube': ion}

            ion_list.append(ion_dict)

        return ion_list

    def fetch_dens(self, nbins=512):
        if self.verbose:
            print('Fetching density cube...')

        dens_list = []
        for n in self.file_nums:
            # if self.verbose:
            #    print(f'Now on file {n}')

            dens_file = f'{self.path_density}/dens_256_out{n}.dat'

            if self.verbose:
                print(f'Now reading in density box from from {dens_file}')

            if not os.path.isfile(dens_file):
                raise FileNotFoundError(dens_file)
            if os.path.isfile(dens_file):
                dens_cube = utils.read_cube(dens_file)
                z = 0
                if n in self.redshifts.keys():
                    z = self.redshifts[n]
                    cube = utils.convert_density(dens_cube, z)

                dens_dict = {'file_n': n,
                            'z': float(z),
                            'cube': cube}
                dens_list.append(dens_dict)

        return dens_list

    def calc_ion_history(self):
        z = []
        xe = []
        for slice in self.ion:
            # if self.verbose:
            #    print(f'Now on file {n}')
            if self.verbose:
                print(f'Calculating ionisation fraction at redshift {slice['z']}')

            cube = slice['cube']
            z.append(slice['z'])
            xe.append(np.mean(cube))

        return z, xe

    def calc_Pee(self, k=None):
        Pee_list = []
        for i, den in enumerate(self.density):
            if self.verbose:
                print(f'Calculating electron power spectrum at redshift {den['z']}')

            file_n = den['file_n']
            z = den['z']
            den_cube = den['cube']
            ion_cube = 0 # this is to increase the scope of the variable
            if self.ion[i]['file_n'] == file_n:
                ion = self.ion[i]
                ion_cube = ion['cube']

            if self.ion[i]['file_n'] != file_n:
                raise Exception("The file numbers don't match")

            delta = (den_cube - np.mean(den_cube)) / np.mean(den_cube)
            ne = ion_cube * (1 + delta)
            ne_overdensity = (ne - np.mean(ne)) / np.mean(ne)

            pk_ne = 0
            bins_ne = 0
            if k is not None:
                if self.verbose:
                    print('Using the k values you asked for')
                pk_ne, bins_ne = get_power(ne_overdensity, 296.0, bins=k)
            if k is None:
                pk_ne, bins_ne = get_power(ne_overdensity, 296.0)
            spectra_dict = {'file_n': file_n,
                            'z': z,
                            'k': bins_ne,
                            'P_k': pk_ne}
            Pee_list.append(spectra_dict)

        return Pee_list
