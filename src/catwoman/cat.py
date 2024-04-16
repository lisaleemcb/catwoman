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
                load_params=False,
                load_Pee=False,
                load_xion=False,
                load_density=False,
                path_sim=None,
                path_params = None,
                path_Pee = None,
                path_xion = None,
                path_density = None):

        print(f'==============================')
        print(f'Loading sim number {sim_n}...')
        print(f'==============================')

        self.sim_n = sim_n
        self.path_sim = path_sim
        self.path_Pee = path_Pee
        self.verbose = verbose

        if load_params:
            if path_params is not None:
                self.path_params = path_params
            else:
                self.path_params = f'{self.path_sim}/simu{self.sim_n}'

        if load_Pee:
            if path_Pee is not None:
                self.path_Pee = path_Pee
            else:
                self.path_Pee = f'{self.path_sim}/simu{self.sim_n}/postprocessing/cubes/ps_ee'


        if load_xion:
            if path_xion is not None:
                self.path_xion = path_xion
            else:
                self.path_xion = f'{self.path_sim}/simu{self.sim_n}/postprocessing/cubes/xion'


        if load_density:
            if path_density is not None:
                self.path_density = path_density
            else:

                self.path_density = f'{self.path_sim}/simu{self.sim_n}/postprocessing/cubes/dens'

        if verbose:
            print('You have told me that data lives in the following places:')
            print('')
            if load_params:
                print(f'params: {self.path_params}')
            if load_Pee:
                print(f'electron power Pee: {self.path_Pee}')
            if load_xion:
                print(f'ionisation cubes: {self.path_xion}')
            if load_density:
                print(f'density cubes: {self.path_density}')
            print('')

        if (load_Pee or load_xion or load_density):
            if verbose:
                print("fetching params since you asked so nicely...")
            self.file_nums = self.gen_filenums()
            self.redshifts = self.fetch_redshifts()

        if load_params:
            self.params = self.fetch_params()

        if load_Pee:
            self.Pee = self.fetch_Pee()

        if load_xion:
            self.xion = self.fetch_xion()

        if load_density:
            self.density = self.fetch_dens()

        print('')
        print("Loaded and ready for science!!")

    def gen_filenums(self):
        file_nums = []
        for filename in os.listdir(f'{self.path_xion}'):
            basename, extensxion = os.path.splitext(filename)

            file_nums.append(basename.split('out')[1])

        return np.sort(file_nums)

    def fetch_params(self):
        if self.verbose:
            print("fetching params since you asked so nicely...")
        fn_params = f'runtime_parameters_simulation_{self.sim_n}_reformatted.txt'

        df = pd.read_csv(f'{self.path_params}/{fn_params}', sep='\t', header=None)
        params = dict(zip(list(df[1]), list(df[0])))
        params['sim_n'] = self.sim_n

        return params

    def fetch_redshifts(self):
        if self.verbose:
            print('Fetching redshifts...')
        # fn_z = f'{self.path_sim}/simu{self.sim_n}/redshift_list.dat'
        fn_z = f'{self.path_sim}/simu{self.sim_n}/postprocessing/cubes/lum/redshift_list.dat'

        redshifts = {}
        with open(fn_z) as f:
                    for line in f:
                        (val, key) = line.split()
                        redshifts[key] = val

        return redshifts

    def fetch_Pee(self, nbins=512):
        if self.verbose:
            print("fetching P_ee since you asked so nicely...")
            print(f'Reading in files from {self.path_Pee}')

        Pee_list = []
        for n in self.file_nums:
            # if self.verbose:
            #     print(f'Now on file {n}')

            Pee_file = f'{self.path_Pee}/powerspectrum_electrons{n}.dat'
            P_ee = (0,0)
            if not os.path.isfile(Pee_file):
                raise FileNotFoundError(Pee_file)
            if os.path.isfile(Pee_file):
                P_ee = np.loadtxt(Pee_file).T

                z = 0
                if n in self.redshifts.keys():
                    z = self.redshifts[n]

                Pee_dict = {'file_n': n,
                                'z': float(z),
                                'k': P_ee[0],
                                'P_k': P_ee[1]}
                Pee_list.append(Pee_dict)

        return Pee_list

    def fetch_xion(self, nbins=512):
        if self.verbose:
            print("fetching xion cubes since you asked so nicely...")

        xion_list = []
        for n in self.file_nums:
            # if self.verbose:
            #     print(f'Now on file {n}')

            xion_file = f'{self.path_xion}/xion_256_out{n}.dat'

            # if self.verbose:
            #     print(f'Now reading in xion box from from {xion_file}')

            if not os.path.isfile(xion_file):
                raise FileNotFoundError(xion_file)
            if os.path.isfile(xion_file):
                xion = utils.read_cube(xion_file)

                z = 0
                if n in self.redshifts.keys():
                    z = self.redshifts[n]

                    xion_dict = {'file_n': n,
                                'z': float(z),
                                'cube': xion}

                    xion_list.append(xion_dict)

        return xion_list

    def fetch_dens(self, nbins=512):
        if self.verbose:
            print("fetching density cubes since you asked so nicely...")

        dens_list = []
        for n in self.file_nums:
            # if self.verbose:
            #    print(f'Now on file {n}')

            dens_file = f'{self.path_density}/dens_256_out{n}.dat'

            # if self.verbose:
            #     print(f'Now reading in density box from from {dens_file}')

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
        if self.verbose:
            print(f"Calculating ionisation history...")

        z = []
        xe = []
        for slice in self.xion:
            # if self.verbose:
            #    print(f'Now on file {n}')
            if self.verbose:
                print(f"Calculating ionisation fraction at redshift {slice['z']}")

            cube = slice['cube']
            z.append(slice['z'])
            xe.append(np.mean(cube))

        return z, xe

    def calc_Pee(self, k=None):
        Pee_list = []
        for i, den in enumerate(self.density):
            if self.verbose:
                print(f"Calculating electron power spectrum at redshift {den['z']}")

            file_n = den['file_n']
            z = den['z']
            den_cube = den['cube']
            xion_cube = 0 # this is to increase the scope of the variable
            if self.xion[i]['file_n'] == file_n:
                xion = self.xion[i]
                xion_cube = xion['cube']

            if self.xion[i]['file_n'] != file_n:
                raise Exception("The file numbers don't match")

            delta = (den_cube - np.mean(den_cube)) / np.mean(den_cube)
            ne = xion_cube * (1 + delta)
            ne_overdensity = (ne - np.mean(ne)) / np.mean(ne)

            pk_ne = 0
            bins_ne = 0
            if k is not None:
                if self.verbose:
                    print('Using the k values you asked for')
                pk_ne, bins_ne = get_power(ne_overdensity, 296.0, bins=k)
            if k is None:
                pk_ne, bins_ne = get_power(ne_overdensity, 296.0, log_bins=True)
            Pee_dict = {'file_n': file_n,
                            'z': z,
                            'k': bins_ne,
                            'P_k': pk_ne}
            Pee_list.append(Pee_dict)

        return Pee_list

    def calc_Pm(self, k=None):
        Pm_list = []
        for i, den in enumerate(self.density):
            if self.verbose:
                print(f"Calculating matter power spectrum at redshift {den['z']}")

            file_n = den['file_n']
            z = den['z']
            den_cube = den['cube']

            delta = (den_cube - np.mean(den_cube)) / np.mean(den_cube)
            pk = 0
            bins = 0
            if k is not None:
                if self.verbose:
                    print('Using the k values you asked for')
                pk, bins = get_power(delta, 296.0, bins=k)
            if k is None:
                pk, __builtins__ = get_power(delta, 296.0)
            Pm_dict = {'file_n': file_n,
                            'z': z,
                            'k': bins,
                            'P_k': pk}
            Pm_list.append(Pm_dict)

        return Pm_list
