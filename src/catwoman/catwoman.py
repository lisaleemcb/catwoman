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
                initialise_spectra=False,
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
                print("Fetching reference files...")
            self.file_nums = self.gen_filenums()
            self.redshift_keys = self.fetch_redshifts()
            self.z = self.redshift_keys.values()

        if load_params:
            self.params = self.fetch_params()

        if load_Pee:
            print('Loading pre-calc Pee...')
            self.Pee = self.fetch_Pee()

        if load_xion:
            self.xion = self.fetch_xion()

        if load_density:
            self.density = self.fetch_dens()

        if initialise_spectra:
            if self.xion:
                if verbose:
                    print('')
                    print('Initialising spectra. This could take a while...')
                    print('')
                self.Pbb = self.calc_Pbb()
                self.Pee = self.calc_Pee()
                self.Pxx = self.calc_Pxx()
                self.k = self.Pee[0]['k']
                self.z, self.xe = self.calc_ion_history()

        print('')
        print("Loaded and ready for science!!")
        print('')

    def gen_filenums(self):
        file_nums = []
        for filename in os.listdir(f'{self.path_xion}'):
            basename, extension = os.path.splitext(filename)

            file_nums.append(basename.split('out')[1])

        return np.sort(file_nums)

    def fetch_params(self):
        if self.verbose:
            print("Fetching params since you asked so nicely...")
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

        if not os.path.isfile(fn_z):
            print('No redshift file with the extension .dat...trying .txt...')
            fn_z = f'{self.path_sim}/simu{self.sim_n}/postprocessing/cubes/lum/redshift_list.txt'

        redshift_keys = {}
        with open(fn_z) as f:
                    for line in f:
                        (val, key) = line.split()
                        redshift_keys[key] = float(val)

        return redshift_keys

    def fetch_Pee(self, nbins=512):
        if self.verbose:
            print("Fetching P_ee since you asked so nicely...")
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
                if n in self.redshift_keys.keys():
                    z = self.redshift_keys[n]

                Pee_dict = {'file_n': n,
                                'z': float(z),
                                'k': P_ee[0],
                                'P_k': P_ee[1]}
                Pee_list.append(Pee_dict)

        return Pee_list

    def fetch_xion(self, nbins=512):
        if self.verbose:
            print("Fetching xion cubes since you asked so nicely...")

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
                if n in self.redshift_keys.keys():
                    z = self.redshift_keys[n]

                    xion_dict = {'file_n': n,
                                'z': float(z),
                                'cube': xion}

                    xion_list.append(xion_dict)

        return xion_list

    def fetch_dens(self, nbins=512):
        if self.verbose:
            print("Fetching density cubes since you asked so nicely...")

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
                if n in self.redshift_keys.keys():
                    z = self.redshift_keys[n]
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

        if self.verbose:
            print('')

        return np.asarray(z), np.asarray(xe)

    def calc_Pee(self, k=None, n_bins=25, log_bins=True):
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

            pk = 0
            bins = 0
            if k is not None:
                if self.verbose:
                    print('Using the k values you asked for')
                pk, bins, var = get_power(ne_overdensity, 296.0, bins=k, get_variance=True)

            if k is None:
                pk, bins, var = get_power(ne_overdensity, 296.0,
                                bins=n_bins,
                                log_bins=log_bins, get_variance=True)

                Pee_dict = {'file_n': file_n,
                                'z': z,
                                'k': bins,
                                'P_k': pk,
                                'var': var}
                Pee_list.append(Pee_dict)

        if self.verbose:
            print('')

        return Pee_list

    def calc_Pbb(self, k=None, n_bins=25, log_bins=True):
        Pbb_list = []
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
                pk, bins, var = get_power(delta, 296.0, bins=k, get_variance=True)
            if k is None:
                pk, bins, var = get_power(delta, 296.0,
                                    bins=n_bins, log_bins=log_bins, get_variance=True)
            Pbb_dict = {'file_n': file_n,
                            'z': z,
                            'k': bins,
                            'P_k': pk,
                            'var': var}
            Pbb_list.append(Pbb_dict)

        if self.verbose:
            print('')

        return Pbb_list

    def calc_Pxx(self, k=None, n_bins=25, log_bins=True):
        Pxx_list = []
        for i, xion in enumerate(self.xion):
            if self.verbose:
                print(f"Calculating ionisation power spectrum at redshift {xion['z']}")

            file_n = xion['file_n']
            z = xion['z']
            xion_cube= xion['cube']

            xion_overdensity = (xion_cube - np.mean(xion_cube)) / np.mean(xion_cube)

            pk = 0
            bins = 0
            if k is not None:
                if self.verbose:
                    print('Using the k values you asked for')
                pk, bins, var = get_power(xion_overdensity, 296.0, bins=k, get_variance=True)

            if k is None:
                pk, bins, var = get_power(xion_cube, 296.0,
                                bins=n_bins,
                                log_bins=log_bins, get_variance=True)

                Pxx_dict = {'file_n': file_n,
                                'z': z,
                                'k': bins,
                                'P_k': pk,
                                'var': var}
                Pxx_list.append(Pxx_dict)

        if self.verbose:
            print('')

        return Pxx_list
