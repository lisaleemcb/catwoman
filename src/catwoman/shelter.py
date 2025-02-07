import logging
from logging import NullHandler
import os
import numpy as np
import pandas as pd

import powerbox as pb
from powerbox import get_power


import catwoman.utils as utils
from catwoman import __version__

__author__ = "Lisa McBride"
__copyright__ = "Lisa McBride"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

class Cat:
    def __init__(self,
                sim_n,
                verbose=False,
                pspec_kwargs=None,
                skip_early=True,
                load_params=True,
                load_spectra=True,
                load_xion_cubes=False,
                load_density_cubes=False,
                reinitialise_spectra=False,
                save_spectra=False,
                just_Pee=True,
                LoReLi_format=False,
                path_sim='/Users/emcbride/kSZ/data/LoReLi',
                path_redshifts='/Users/emcbride/kSZ/data/LoReLi_summaries/redshift_list.dat',
                path_params = None,
                path_spectra=None,
                path_Pee = None,
                path_ion=None,
                path_xion = None,
                path_density = None,
                k=np.array([0.021227 , 0.0300195, 0.0392038, 0.0497301, 0.0659062, 0.0834707,
                            0.1051155, 0.132222 , 0.1692068, 0.2162669, 0.2754583, 0.3508495,
                            0.4475445, 0.5709843, 0.7276318, 0.9274267, 1.1823019, 1.5068212,
                            1.9204267, 2.4476898]),
                debug=False):
    
        self.sim_n = sim_n
        self.path_sim = path_sim
        self.path_redshifts = path_redshifts
        self.path_Pee = path_Pee
        self.verbose = verbose
        self.skip_early = skip_early
        self.debug = debug

        self.box_size= 296.0 # Mpc
        self.k_res = ((2 * np.pi) / self.box_size, (2 * np.pi * 256) / self.box_size / 2)

        if self.verbose:
            print(f'==============================')
            print(f'Loading sim number {sim_n}...')
            print(f'==============================')

        if load_params:
            if path_params is not None:
                self.path_params = path_params
            else:
                self.path_params = f'{self.path_sim}/simu{self.sim_n}'

        if path_spectra is not None:
            self.path_spectra = path_spectra
        else:
            self.path_spectra = f'{self.path_sim}/simu{self.sim_n}/spectra'
            
        if not os.path.exists(self.path_spectra):
                os.makedirs(self.path_spectra)

        if LoReLi_format:
            self.Pee_spectra_path  = f'{self.path_sim}/simu{self.sim_n}/postprocessing/cubes/ps_dtb'
        else: 
            self.Pee_spectra_path = f'{self.path_spectra}/simu{self.sim_n}_Pee_spectra.npz'

        if not just_Pee:
            self.Pbb__spectra_path = f'{self.path_spectra}/simu{self.sim_n}_Pbb_spectra.npz'
            self.Pxx_spectra_path = f'{self.path_spectra}/simu{self.sim_n}_Pxx_spectra.npz'

        if path_xion is not None:
            self.path_xion = path_xion
        else:
            self.path_xion = f'{self.path_sim}/simu{self.sim_n}/postprocessing/cubes/xion'

        if path_density is not None:
            self.path_density = path_density
        else:
            self.path_density = f'{self.path_sim}/simu{self.sim_n}/postprocessing/cubes/dens'

        if verbose:
            print('You have told me that data lives in the following places:')
            print(f'\tredshift list: {self.path_redshifts}')
            if load_params:
                print(f'\tparams: {self.path_params}')
            if load_xion_cubes:
                print(f'\tionisation cubes: {self.path_xion}')
            if load_density_cubes:
                print(f'\tdensity cubes: {self.path_density}')
            print('')

        if (load_xion_cubes or load_density_cubes or reinitialise_spectra):
            if verbose:
                print("Fetching reference files...")
            self.file_nums = self.gen_filenums()
            self.redshift_keys = self.fetch_redshifts()
            self.z = self.redshift_keys.values()


        fn_params = f'runtime_parameters_simulation_{self.sim_n}_reformatted.txt'
        if (load_params or reinitialise_spectra):
            if self.verbose:
                print("Fetching params...")
            self.params = utils.read_params(f'{self.path_params}/{fn_params}')

        if (load_xion_cubes or reinitialise_spectra):
            self.xion = self.load_xion_cubes()

        if (load_density_cubes or reinitialise_spectra):
            self.density = self.load_density_cubes()

        if reinitialise_spectra:
            if pspec_kwargs is None:
                self.pspec_kwargs = {'bins': np.geomspace(self.k_res[0], self.k_res[1], 21),
                                    'log_bins': True,
                                    'get_variance': False,
                                    'bin_ave': True}
            else:
                self.pspec_kwargs = pspec_kwargs
            
            if self.verbose:
                print('')
                print(f'Simulation runs from z={max(list(self.z))} to z={min(list(self.z))}')
                print(f'Power spectrum settings:')
                print(f'\t{self.pspec_kwargs}')

            if self.xion: # this just checks that the data cubes exist
                if verbose:
                    print('')
                    print('Initialising spectra since you asked so nicely! But this could take a while...')
                    print('')
                self.Pee_dict = self.calc_Pee()
                if not just_Pee:
                    self.Pbb_dict = self.calc_Pbb()
                    self.Pxx_dict = self.calc_Pxx()

                self.z, self.xe = self.calc_ion_history()
                self.k = self.Pee_dict[0]['k']
                self.Pee = utils.unpack_data(self.Pee_dict)
                if not just_Pee:
                    self.Pbb = utils.unpack_data(self.Pbb_dict)
                    self.Pxx = utils.unpack_data(self.Pxx_dict)

                if save_spectra:
                    if verbose:
                        print('Saving power spectra...')
                        print(f'   Pee path: {self.Pee_spectra_path}')

                    np.savez(self.Pee_spectra_path, k=self.k, z=self.z, xe=self.xe, Pk=self.Pee)
                    if not just_Pee:
                        np.savez(self.Pbb_spectra_path, k=self.k, z=self.z, xe=self.xe, Pk=self.Pbb)
                        np.savez(self.Pxx_spectra_path, k=self.k, z=self.z, xe=self.xe, Pk=self.Pxx)

                if self.skip_early:
                    self.skip = utils.find_index(self.xe) # to pick out monotonically increasing xe only
                    if np.isnan(self.skip):
                        self.skip = 0
                else:
                    self.skip = 0

                self.z = self.z[self.skip:]
                self.xe = self.xe[self.skip:]
                self.Pee = self.Pee[self.skip:]
                if not just_Pee:
                    self.Pbb = self.Pbb[self.skip:]
                    self.Pxx =self.Pxx[self.skip:]

                if verbose:
                    print('')
                    print("Loaded and ready for science!!")
                    print('')

            if not self.xion:
                print('')
                print('No spectra here, only danger!!!')
                print('')

        elif not reinitialise_spectra:
            if load_spectra:
                if not LoReLi_format:
                    if self.verbose:
                        print('')
                        print('Loading precalculated spectra. If you would like fresh spectra, rerun with reinitialise_spectra=True')
                        print('')
                        
                    if not os.path.exists(self.Pee_spectra_path):
                            raise FileNotFoundError(f"The file '{self.Pee_spectra_path}' does not exist.\nRerun with reinitialise_spectra=True and save_spectra=True.")
                    
                    if not just_Pee:
                        if not os.path.exists(self.Pbb_spectra_path):
                                raise FileNotFoundError(f"The file '{self.Pbb_spectra_path}' does not exist. \nRerun with reinitialise_spectra=True and save_spectra=True.")
                        
                        if not os.path.exists(self.Pxx_spectra_path):
                                raise FileNotFoundError(f"The file '{self.Pxx_spectra_path}' does not exist. \nRerun with reinitialise_spectra=True and save_spectra=True.")
                        
                    Pee_file = np.load(self.Pee_spectra_path)
                    if not just_Pee:
                        Pbb_file = np.load(self.Pbb_spectra_path)
                        Pxx_file = np.load(self.Pxx_spectra_path)
                    
                    self.k = Pee_file['k']
                    self.xe = Pee_file['xe']
                    if self.skip_early:
                        self.skip = utils.find_index(self.xe) # to pick out monotonically increasing xe only
                    else:
                        self.skip = 0
                    self.z = Pee_file['z'][self.skip:]
                    self.xe = self.xe[self.skip:]
                    self.Pee = Pee_file['Pk'][self.skip:]
                    if not just_Pee:
                        self.Pbb = Pbb_file['Pk'][self.skip:]
                        self.Pxx = Pxx_file['Pk'][self.skip:]
                
                elif LoReLi_format:
                    if self.verbose:
                        print('')
                        print('Loading precalculated spectra (LoReLi format). If you would like fresh spectra, rerun with reinitialise_spectra=True')
                        print('')
                        print("Fetching reference files as required for LoReLi format...")

                    # file_nums = []
                    # for filename in os.listdir(f'{self.path_xion}'):
                    #     basename, extension = os.path.splitext(filename)

                    #     file_nums.append(basename.split('out')[1])

                    # return np.sort(file_nums)
                
                    # self.file_nums = self.gen_filenums()
                    self.redshift_keys = self.fetch_redshifts()

                    ion_histories = np.load(path_ion, allow_pickle=True)
                    ion_histories = ion_histories['arr_0'].item()

                    self.z = ion_histories[self.sim_n]['z']
                    self.xe = ion_histories[self.sim_n]['xe']

                    if self.debug:
                        print(f'ionisation fraction is:')
                        print(f'\t xe={self.xe}')
                        
                    spectra = []
                    z_indices = []
                 #   print(f'redshift are: \n \t{self.z}')
                    self.which_keys = []
                    for key in self.redshift_keys.keys():
                        fn = f'{self.Pee_spectra_path}/powerspectrum_electrons{key}_logbins.dat'
                        #print(fn)
                        if os.path.isfile(fn):
                            keyz_rounded = utils.round_sig_figs(self.redshift_keys[key])
                            match = np.where(np.isclose(self.z, keyz_rounded, rtol=1e-3))[0]

                            if self.debug:
                                print(f'key: {key}, redshift: {self.redshift_keys[key]}, zrounded: {keyz_rounded}')
                                print(f'match: {match}')
                        
                            if len(match) > 0:
                                self.which_keys.append(key)
                                index = match[0]
                                if self.debug:
                                    print(f'keyz: {keyz_rounded}, z_xe: {self.z[index]}')
                                    print()
                                z_indices.append(index)
                          #      print(f'Log has redshift {self.redshift_keys[key]}')
                                s = np.genfromtxt(fn)
                                if len(s.shape) == 2:
                                    spectra.append(s)
                                   # print(f'key: {key}, spectra: {s.shape}')
                            else:
                                if self.debug:
                                    print('no match!')
                                    print()

                    if spectra:
                        self.k = spectra[0][:,0]
                        
                        #print(f'Indices: {z_indices}')
                        self.z = self.z[z_indices].flatten()
                        self.xe = self.xe[z_indices].flatten()

                        if k is not None:
                            self.k = k

                        self.Pee = np.zeros((len(spectra), self.k.size))

                        #print(f'Pee shape is {self.Pee.shape} but the length of spectra is {len(spectra)}')
                        for i in range(len(spectra)):
                            self.Pee[i] = spectra[i][:,1].flatten()

                        if self.skip_early:
                            self.skip = utils.find_index(self.xe) # to pick out monotonically increasing xe only
                        else:
                            self.skip = 0
                        
                        self.z = self.z[self.skip:]
                        self.xe = self.xe[self.skip:]
                        self.Pee = self.Pee[self.skip:]
            
                        if not just_Pee:
                            pass
                        # self.Pbb = Pbb_file['Pk'][self.skip:]
                        # self.Pxx = Pxx_file['Pk'][self.skip:]

                    if verbose:
                        print('')
                        print(f"Simulation {self.sim_n} loaded and ready for science!!")
                        print('')

                    if not spectra:
                        self.Pee = np.nan
                        print(f'Sim  {self.sim_n} is loaded but there is no data!')

    def gen_filenums(self):
        file_nums = []
        for filename in os.listdir(f'{self.path_xion}'):
            basename, extension = os.path.splitext(filename)

            file_nums.append(basename.split('out')[1])

        return np.sort(file_nums)

    def fetch_params(self):
        # currently obsolete
        if self.verbose:
            print("Fetching params...")
        fn_params = f'runtime_parameters_simulation_{self.sim_n}_reformatted.txt'

        df = pd.read_csv(f'{self.path_params}/{fn_params}', sep='\t', header=None)
        params = dict(zip(list(df[1]), list(df[0])))
        params['sim_n'] = self.sim_n

        return params
    
    def Delta2(self):
        return (self.k**3 / (2.0 * np.pi**2)) * self.Pee

    def fetch_redshifts(self):
        if self.verbose:
            print('Fetching redshifts...')
        # fn_z = f'{self.path_sim}/simu{self.sim_n}/redshift_list.dat'
        if self.verbose:
            print(f'Trying {self.path_redshifts}')

        redshift_keys = {}
        with open(self.path_redshifts) as f:
                    for line in f:
                        (val, key) = line.split()
                        redshift_keys[key] = float(val)

        return redshift_keys

    def load_xion_cubes(self, nbins=512):
        if self.verbose:
            print("Fetching xion cubes...")

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

    def load_density_cubes(self, nbins=512):
        if self.verbose:
            print("Fetching density cubes...")

        density_list = []
        for n in self.file_nums:
            # if self.verbose:
            #    print(f'Now on file {n}')

            density_file = f'{self.path_density}/dens_256_out{n}.dat'

            # if self.verbose:
            #     print(f'Now reading in density box from from {density_file}')

            if not os.path.isfile(density_file):
                raise FileNotFoundError(density_file)
            if os.path.isfile(density_file):
                density_cube = utils.read_cube(density_file)
                z = 0
                if n in self.redshift_keys.keys():
                    z = self.redshift_keys[n]
                    cube = utils.convert_density(density_cube, z)

                    density_dict = {'file_n': n,
                                'z': float(z),
                                'cube': cube}
                    density_list.append(density_dict)

        return density_list

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

            pk = 0
            bins = 0
            if k is not None:
                if self.verbose:
                    print('Using the k values you asked for')
                pk, bins = get_power(ne_overdensity, self.box_size, **self.pspec_kwargs)

            if k is None:
                pk, bins  = get_power(ne_overdensity, self.box_size,
                               **self.pspec_kwargs)

            Pee_dict = {'file_n': file_n,
                            'z': z,
                            'k': bins,
                            'P_k': pk}
            Pee_list.append(Pee_dict)

        if self.verbose:
            print('')

        return Pee_list

    def calc_Pbb(self, k=None):
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
                pk, bins = get_power(delta, self.box_size, **self.pspec_kwargs)

            if k is None:
                pk, bins  = get_power(delta, self.box_size,
                               **self.pspec_kwargs)
            Pbb_dict = {'file_n': file_n,
                            'z': z,
                            'k': bins,
                            'P_k': pk}
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
                pk, bins = get_power(xion_overdensity, self.box_size, **self.pspec_kwargs)

            if k is None:
                pk, bins  = get_power(xion_overdensity, self.box_size,
                               **self.pspec_kwargs)

                Pxx_dict = {'file_n': file_n,
                                'z': z,
                                'k': bins,
                                'P_k': pk}
                Pxx_list.append(Pxx_dict)

        if self.verbose:
            print('')

        return Pxx_list

# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration")
    parser.add_argument(
        "--version",
        action="version",
        version=f"catwoman {__version__}",
    )
    parser.add_argument(dest="n", help="n-th Fibonacci number", type=int, metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("skiping crazy calculations...")
    print(f"The {args.n}-th Fibonacci number is {fib(args.n)}")
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m catwoman.skeleton 42
    #
    run()
