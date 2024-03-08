from logging import NullHandler
import os
import numpy as np
import pandas as pd

import catwoman.utils as utils

class Cat:
    def __init__(self,
                sim_n,
                verbose=False,
                path_sim='/Users/emcbride/kSZ/data',
               # path_params = f'Pee_spectra_LoReLi/formatted/',
               # path_Pee = 'Pee_spectra_LoReLi/raw/',
                ):

        print(f'Loading sim number {sim_n}...')

        self.sim_n = sim_n
        self.path_sim = path_sim
        self.path_params = f'Pee_spectra_LoReLi/formatted/simu{self.sim_n}'
        self.path_Pee = f'Pee_spectra_LoReLi/raw/simu{self.sim_n}'

        self.verbose = verbose

        self.file_nums = self.gen_filenums()
        self.params = self.fetch_params()
        self.redshifts = self.fetch_redshifts()
        self.spectra = self.fetch_spectra()
        self.ion_cube = self.fetch_ion()
        self.dens_cube = self.fetch_dens()

        print("Loaded and ready for science!!")

    def gen_filenums(self):
        file_nums = []
        for filename in os.listdir(f'{self.path_sim}/{self.path_Pee}/postprocessing/cubes/ps_dtb'):
            basename, extension = os.path.splitext(filename)
            file_nums.append(basename.split('electrons')[1])

        return np.sort(file_nums)

    def fetch_params(self):
        if self.verbose:
            print('Fetching simulations parameters...')
        fn_params = f'runtime_parameters_simulation_{self.sim_n}_reformatted.txt'

        if self.verbose:
            print(f'Now reading in params from {self.path_sim}/{self.path_params}/{fn_params}')

        df = pd.read_csv(f'{self.path_sim}/{self.path_params}/{fn_params}', sep='\t', header=None)
        params = dict(zip(list(df[1]), list(df[0])))
        params['sim_n'] = self.sim_n

        return params

    def fetch_redshifts(self):
        if self.verbose:
            print('Fetching redshifts...')
        fn_z = f'{self.path_sim}/{self.path_Pee}/redshift_list.dat'

        redshifts = {}
        with open(fn_z) as f:
                    for line in f:
                        (val, key) = line.split()
                        redshifts[key] = val

        return redshifts

    def fetch_spectra(self, nbins=512):
        if self.verbose:
            print('Fetching Pee spectra...')
            print(f'Now reading in params from {self.path_sim}/{self.path_Pee}')

        Pee_list = []
        for n in self.file_nums:
            if self.verbose:
                print(f'Now on file {n}')

            ion_file = f'{self.path_sim}/xion/xion_256_out{n}.dat'
            ion_cube = 0
            if os.path.isfile(ion_file):
                ion_cube = utils.read_cube(ion_file)

            dens_file = f'{self.path_sim}/dens/dens_256_out{n}.dat'
            dens_cube = 0
            if os.path.isfile(dens_file):
                dens_cube = utils.read_cube(dens_file)

            Pee_file = f'{self.path_sim}/{self.path_Pee}/postprocessing/cubes/ps_dtb/powerspectrum_electrons{n}.dat'
            P_ee = (0,0)
            if os.path.isfile(Pee_file):
                P_ee = np.loadtxt(Pee_file).T

                z = 0
                if n in self.redshifts.keys():
                    z = self.redshifts[n]

            spectra_dict = {'file_n': n,
                            'z': float(z),
                          #  'dens_cube': dens_cube,
                          #  'ion_cube': ion_cube,
                            'k': P_ee[0],
                            'P_k': P_ee[1]}
            Pee_list.append(spectra_dict)

        return Pee_list

    def fetch_ion(self, nbins=512):
        if self.verbose:
            print('Fetching ion cube...')

        ion_list = []
        for n in self.file_nums:
            if self.verbose:
                print(f'Now on file {n}')

            ion_file = f'{self.path_sim}/xion/xion_256_out{n}.dat'
            ion_cube = 0

            if self.verbose:
                print(f'Now reading in ion box from from {ion_file}')
            if os.path.isfile(ion_file):
                ion_cube = utils.read_cube(ion_file)

            z = 0
            if n in self.redshifts.keys():
                z = self.redshifts[n]

            ion_dict = {'file_n': n,
                        'z': float(z),
                        'ion_cube': ion_cube}

            ion_list.append(ion_dict)

        return ion_list

    def fetch_dens(self, nbins=512):
        if self.verbose:
            print('Fetching density cube...')

        dens_list = []
        for n in self.file_nums:
            if self.verbose:
               print(f'Now on file {n}')

            dens_file = f'{self.path_sim}/xion/xion_256_out{n}.dat'
            dens_cube = 0

            if self.verbose:
                print(f'Now reading in ion box from from {dens_file}')
            if os.path.isfile(dens_file):
                dens_cube = utils.read_cube(dens_file)
                z = 0
                if n in self.redshifts.keys():
                    z = self.redshifts[n]
                    dens_cube = utils.convert_density(dens_cube, z)

                dens_dict = {'file_n': n,
                            'z': float(z),
                            'dens_cube': dens_cube}
                dens_list.append(dens_dict)

        return dens_list

    def calc_ion_history(self):
        ion_file = f'{self.path_sim}/xion/xion_256_out{n}.dat'
        ion_cube = read_cube(ion_file)

        return 0
