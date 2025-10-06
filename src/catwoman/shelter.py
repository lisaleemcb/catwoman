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
    """ """

    def __init__(
        self,
        sim_n,
        verbose=False,
        pspec_kwargs=None,
        skip_early=True,
        load_params=False,
        load_spectra=True,
        load_xion_cubes=False,
        load_density_cubes=False,
        load_T21cm_cubes=False,
        reinitialise_spectra=False,
        use_LoReLi_xe=False,
        save_spectra=False,
        just_Pee=True,
        LoReLi_format=False,
        redshifts_fn="metadata/redshift_list.dat",
        ionhistories_fn="metadata/ion_histories_full.npz",
        base_dir="/Users/emcbride/Datasets/LoReLi",
        path_sim="Datasets/LoReLi",
        path_params="metadata/param_files",
        path_spectra="ps_ee",
        path_xion_cubes=None,
        path_density_cubes=None,
        k=np.array(
            [
                0.021227,
                0.0300195,
                0.0392038,
                0.0497301,
                0.0659062,
                0.0834707,
                0.1051155,
                0.132222,
                0.1692068,
                0.2162669,
                0.2754583,
                0.3508495,
                0.4475445,
                0.5709843,
                0.7276318,
                0.9274267,
                1.1823019,
                1.5068212,
                1.9204267,
                2.4476898,
            ]
        ),
        debug=False,
    ):
        self.sim_n = sim_n
        self.base_dir = base_dir
        self.path_sim = path_sim
        self.redshifts_fn = redshifts_fn
        self.path_spectra = path_spectra
        self.use_LoReLi_xe = use_LoReLi_xe
        self.verbose = verbose
        self.skip_early = skip_early
        self.debug = debug

        self.box_size = 296.0  # Mpc
        self.k_res = (
            (2 * np.pi) / self.box_size,
            (2 * np.pi * 256) / self.box_size / 2,
        )

        self.density = None
        self.xion = None
        self.T21cm = None

        if self.verbose:
            print(f"==============================")
            print(f"Loading sim number {sim_n}...")
            print(f"==============================")

        if load_spectra and reinitialise_spectra:
            raise ValueError(
                "Both load_spectra and reinitialise_spectra can be set to true."
            )

        if load_params:
            if path_params is not None:
                self.path_params = path_params
            else:
                self.path_params = f"{self.path_sim}/simu{self.sim_n}"

            if self.base_dir:
                self.path_params = f"{self.base_dir}/{self.path_params}"

        if path_sim is not None:
            self.path_sim = path_sim
        else:
            self.path_sim = f"{self.base_dir}/simu{self.sim_n}/spectra"

        if path_xion_cubes is not None:
            self.path_xion_cubes = path_xion_cubes
        else:
            self.path_xion_cubes = (
                f"{self.path_sim}/simu{self.sim_n}/postprocessing/cubes/xion"
            )

        if path_density_cubes is not None:
            self.path_density_cubes = path_density_cubes
        else:
            self.path_density_cubes = (
                f"{self.path_sim}/simu{self.sim_n}/postprocessing/cubes/dens"
            )

        self.path_T21cm_cubes = (
            f"{self.path_sim}/simu{self.sim_n}/postprocessing/cubes/dtb"
        )

        if self.base_dir:
            self.redshifts_fn = f"{self.base_dir}/{self.redshifts_fn}"

        if self.verbose:
            if load_params:
                print("Fetching params from:")
                print(f"\tparams: {self.path_params}")
                print("")
            if load_xion_cubes:
                print("Fetching xion cubes from:")
                print(f"\tionisation cubes: {self.path_xion_cubes}")
                print("")
            if load_density_cubes:
                print("Fetching density cubes from:")
                print(f"\tdensity cubes: {self.path_density_cubes}")
                print("")

        if (
            load_xion_cubes
            or load_density_cubes
            or load_T21cm_cubes
            or reinitialise_spectra
        ):
            if self.verbose:
                print("Fetching reference files...")
            self.file_nums = self.gen_filenums()
            self.redshift_keys = self.fetch_redshifts()
            self.z = self.redshift_keys.values()

        fn_params = f"runtime_parameters_simulation_{self.sim_n}_reformatted.txt"
        if load_params:
            if self.verbose:
                print("Fetching params...")
            self.params = utils.read_params(f"{self.path_params}/{fn_params}")

        if load_xion_cubes or reinitialise_spectra:
            self.xion, self.z = self.load_cubes(
                self.path_xion_cubes, "xion_256_out", type="xion"
            )

        if load_density_cubes or reinitialise_spectra:
            self.density, self.z = self.load_cubes(
                self.path_density_cubes, "dens_256_out", type="density"
            )

        if load_T21cm_cubes:
            self.T21cm, self.z = self.load_cubes(
                self.path_T21cm_cubes, "dtb_tp_hi_256_nocorrection_out", type="T21cm"
            )

        if self.xion is not None:  # this just checks that the data cubes exist
            print("Calculating ionisation history...")
            self.xe = np.mean(self.xion, axis=(1, 2, 3))
        if reinitialise_spectra:
            if self.verbose:
                print("")
                print(
                    "Initialising spectra since you asked so nicely! But this could take a while..."
                )
                print("")

            if pspec_kwargs is None:
                self.pspec_kwargs = {
                    "bins": np.geomspace(self.k_res[0], self.k_res[1], 21),
                    "log_bins": True,
                    "get_variance": False,
                    "bin_ave": True,
                }
            else:
                self.pspec_kwargs = pspec_kwargs

            if self.verbose:
                print("")
                print(
                    f"Simulation runs from z={max(list(self.z))} to z={min(list(self.z))}"
                )
                print(f"Power spectrum settings:")
                print(f"\t{self.pspec_kwargs}")

            if self.xion is not None:  # this just checks that the data cubes exist
                if self.verbose:
                    print("")
                    print(
                        "Initialising spectra since you asked so nicely! But this could take a while..."
                    )
                    print("")
                self.Pee, self.k = self.calculate_power(self.density, type="electron")
                if not just_Pee:
                    self.Pbb, self.k = self.calculate_power(
                        self.density, type="density"
                    )
                    self.Pxx, self.k = self.calculate_power(self.xion, type="xion")

                if save_spectra:
                    self.Pee_spectra_path = (
                        f"{path_spectra}/simu{self.sim_n}_Pee_spectra.npz"
                    )

                    if self.verbose:
                        print("Saving power spectra...")
                        print(f"   Pee path: {self.Pee_spectra_path}")

                    np.savez(
                        self.Pee_spectra_path,
                        k=self.k,
                        z=self.z,
                        xe=self.xe,
                        Pk=self.Pee,
                    )
                    if not just_Pee:
                        np.savez(
                            self.Pbb_spectra_path,
                            k=self.k,
                            z=self.z,
                            xe=self.xe,
                            Pk=self.Pbb,
                        )
                        np.savez(
                            self.Pxx_spectra_path,
                            k=self.k,
                            z=self.z,
                            xe=self.xe,
                            Pk=self.Pxx,
                        )

                if self.skip_early:
                    self.skip = utils.find_index(
                        self.xe
                    )  # to pick out monotonically increasing xe only
                    if np.isnan(self.skip):
                        self.skip = 0
                else:
                    self.skip = 0

                self.z = self.z[self.skip :]
                self.xe = self.xe[self.skip :]
                self.Pee = self.Pee[self.skip :]
                if not just_Pee:
                    self.Pbb = self.Pbb[self.skip :]
                    self.Pxx = self.Pxx[self.skip :]

                if self.verbose:
                    print("")
                    print("Loaded and ready for science!!")
                    print("")

            if not self.xion:
                print("")
                print("No spectra here, only danger!!!")
                print("")

        if load_spectra:
            if LoReLi_format:
                self.Pee_spectra_path = (
                    f"{self.path_spectra}/simu{self.sim_n}/postprocessing/cubes/ps_dtb"
                )
            else:
                self.Pee_spectra_path = (
                    f"{self.path_spectra}/Pee/simu{self.sim_n}_Pee_spectra.npz"
                )

            if self.base_dir:
                self.Pee_spectra_path = f"{self.base_dir}/{self.Pee_spectra_path}"

            if not os.path.exists(self.Pee_spectra_path):
                raise FileNotFoundError(
                    f"The file '{self.Pee_spectra_path}' does not exist.{os.linesep}Rerun with reinitialise_spectra=True and save_spectra=True."
                )

            if not just_Pee:
                self.Pbb_spectra_path = (
                    f"{self.path_spectra}/Pbb/simu{self.sim_n}_Pbb_spectra.npz"
                )
                self.Pxx_spectra_path = (
                    f"{self.path_spectra}/Pxx/simu{self.sim_n}_Pxx_spectra.npz"
                )

                if self.base_dir:
                    self.Pbb_spectra_path = f"{self.base_dir}/{self.Pbb_spectra_path}"
                    self.Pxx_spectra_path = f"{self.base_dir}/{self.Pxx_spectra_path}"

                    if not os.path.exists(self.Pbb_spectra_path):
                        raise FileNotFoundError(
                            f"The file '{self.Pbb_spectra_path}' does not exist. \nRerun with reinitialise_spectra=True and save_spectra=True."
                        )

                    if not os.path.exists(self.Pxx_spectra_path):
                        raise FileNotFoundError(
                            f"The file '{self.Pxx_spectra_path}' does not exist. \nRerun with reinitialise_spectra=True and save_spectra=True."
                        )

            if self.verbose:
                print("Loading the follow spectra from:")
                print(f"\tPee: {self.Pee_spectra_path}")
                if not just_Pee:
                    print(f"\tPbb: {self.Pbb_spectra_path}")
                    print(f"\tPxx: {self.Pxx_spectra_path}")

            if not LoReLi_format:
                if self.verbose:
                    print("")
                    print(
                        "Loading precalculated spectra (from cubes). If you would like fresh spectra, rerun with reinitialise_spectra=True"
                    )
                    print("")

                Pee_file = np.load(self.Pee_spectra_path)
                if not just_Pee:
                    Pbb_file = np.load(self.Pbb_spectra_path)
                    Pxx_file = np.load(self.Pxx_spectra_path)

                self.k = Pee_file["k"]
                self.xe = Pee_file["xe"]
                if self.skip_early:
                    self.skip = utils.find_index(
                        self.xe
                    )  # to pick out monotonically increasing xe only
                else:
                    self.skip = 0

                self.z = Pee_file["z"][self.skip :]
                self.xe = self.xe[self.skip :]
                self.Pee = Pee_file["Pk"][self.skip :]

                if self.use_LoReLi_xe:
                    if self.verbose:
                        print("Subbing in LoReLi ionisation history...")
                    if self.base_dir:
                        ionhistories_fn = f"{self.base_dir}/{ionhistories_fn}"

                    ion_histories = np.load(ionhistories_fn, allow_pickle=True)
                    ion_histories = ion_histories["arr_0"].item()

                    z_LoReLi = ion_histories[self.sim_n]["z"]
                    xe_LoReLi = ion_histories[self.sim_n]["xe"]

                    tol = 0.01  # tolerance
                    diff = np.abs(z_LoReLi[:, None] - self.z[None, :])
                    matches = np.where(diff <= tol)

                    # matches[0] are indices in `a`, matches[1] are corresponding indices in `b`

                    self.z = z_LoReLi[matches[0]]
                    self.xe = xe_LoReLi[matches[0]]
                    self.Pee = self.Pee[matches[1], :]

                    # self.z = self.z[z_indices].flatten()
                    # self.xe = self.xe[z_indices].flatten()
                    # self.Pee = self.Pee[z_indices,:]

                if not just_Pee:
                    self.Pbb = Pbb_file["Pk"][self.skip :]
                    self.Pxx = Pxx_file["Pk"][self.skip :]

            elif LoReLi_format:
                if self.verbose:
                    print("")
                    print(
                        "Loading precalculated spectra (LoReLi format). If you would like fresh spectra, rerun with reinitialise_spectra=True"
                    )
                    print("")

                self.redshift_keys = self.fetch_redshifts()

                if self.base_dir:
                    ionhistories_fn = f"{self.base_dir}/{ionhistories_fn}"

                if self.verbose:
                    print("Fetching ionisation history from:")
                    print(f"\t {ionhistories_fn}")

                ion_histories = np.load(ionhistories_fn, allow_pickle=True)
                ion_histories = ion_histories["arr_0"].item()

                self.z = ion_histories[self.sim_n]["z"]
                self.xe = ion_histories[self.sim_n]["xe"]

                if self.debug:
                    print(f"ionisation fraction is:")
                    print(f"\t xe={self.xe}")

                spectra = []
                z_indices = []
                #   print(f'redshift are: \n \t{self.z}')
                self.which_keys = []
                for key in self.redshift_keys.keys():
                    fn = f"{self.Pee_spectra_path}/powerspectrum_electrons{key}_logbins.dat"
                    # print(fn)
                    if os.path.isfile(fn):
                        keyz_rounded = utils.round_sig_figs(self.redshift_keys[key])
                        match = np.where(np.isclose(self.z, keyz_rounded, rtol=1e-3))[0]

                        if self.debug:
                            print(
                                f"key: {key}, redshift: {self.redshift_keys[key]}, zrounded: {keyz_rounded}"
                            )
                            print(f"match: {match}")

                        if len(match) > 0:
                            self.which_keys.append(key)
                            index = match[0]
                            if self.debug:
                                print(f"keyz: {keyz_rounded}, z_xe: {self.z[index]}")
                                print()
                            z_indices.append(index)
                            #      print(f'Log has redshift {self.redshift_keys[key]}')
                            s = np.genfromtxt(fn)
                            if len(s.shape) == 2:
                                spectra.append(s)
                                # print(f'key: {key}, spectra: {s.shape}')
                        else:
                            if self.debug:
                                print("no match!")
                                print()

                if spectra:
                    self.k = spectra[0][:, 0]

                    # print(f'Indices: {z_indices}')
                    self.z = self.z[z_indices].flatten()
                    self.xe = self.xe[z_indices].flatten()

                    if k is not None:
                        self.k = k

                    self.Pee = np.zeros((len(spectra), self.k.size))

                    # print(f'Pee shape is {self.Pee.shape} but the length of spectra is {len(spectra)}')
                    for i in range(len(spectra)):
                        self.Pee[i] = spectra[i][:, 1].flatten()

                    if self.skip_early:
                        self.skip = utils.find_index(
                            self.xe
                        )  # to pick out monotonically increasing xe only
                    else:
                        self.skip = 0

                    self.z = self.z[self.skip :]
                    self.xe = self.xe[self.skip :]
                    self.Pee = self.Pee[self.skip :]

                    if not just_Pee:
                        pass
                    # self.Pbb = Pbb_file['Pk'][self.skip:]
                    # self.Pxx = Pxx_file['Pk'][self.skip:]

                if not spectra:
                    self.Pee = np.nan
                    print(f"Sim  {self.sim_n} is loaded but there is no data!")

        if self.verbose:
            print("")
            print(f"Simulation {self.sim_n} loaded and ready for science!!")
            print("")

    def gen_filenums(self):
        """ """
        file_nums = []
        for filename in os.listdir(f"{self.path_xion_cubes}"):
            basename, extension = os.path.splitext(filename)

            file_nums.append(basename.split("out")[1])

        return np.sort(file_nums)

    def fetch_params(self):
        """ """
        # currently obsolete
        if self.verbose:
            print("Fetching params...")
        fn_params = f"runtime_parameters_simulation_{self.sim_n}_reformatted.txt"

        df = pd.read_csv(f"{self.path_params}/{fn_params}", sep="\t", header=None)
        params = dict(zip(list(df[1]), list(df[0])))
        params["sim_n"] = self.sim_n

        return params

    def Delta2(self):
        """ """
        return (self.k**3 / (2.0 * np.pi**2)) * self.Pee

    def fetch_redshifts(self):
        """ """
        if self.verbose:
            print("Fetching redshifts from:")
            print(f"\t {self.redshifts_fn}")

        redshift_keys = {}
        with open(self.redshifts_fn) as f:
            for line in f:
                (val, key) = line.split()
                redshift_keys[key] = float(val)

        return redshift_keys

    def load_cubes(self, fn, ext, type="simulation", nbins=512):
        """

        :param fn:
        :param ext:
        :param nbins:  (Default value = 512)

        """
        if self.verbose:
            print(f"Fetching {type} cubes...")

        cubes_list = []
        for n in self.file_nums:
            # if self.verbose:
            #    print(f'Now on file {n}')

            # if self.verbose:
            #     print(f'Now reading in density box from from {filename}')
            #
            filename = f"{fn}/{ext}{n}.dat"

            if not os.path.isfile(filename):
                raise FileNotFoundError(filename)
            if os.path.isfile(filename):
                cube = utils.read_cube(filename)
                z = 0
                if n in self.redshift_keys.keys():
                    z = self.redshift_keys[n]
                    if type == "density":
                        cube = utils.convert_density(cube, z)

                    dict = {"file_n": n, "z": float(z), "cube": cube}
                    cubes_list.append(dict)

        z = np.zeros(len(cubes_list))
        cubes = np.zeros((len(cubes_list), *cubes_list[0]["cube"].shape))
        for i, c in enumerate(cubes_list):
            z[i] = c["z"]
            cubes[i] = c["cube"]

        return cubes, z

    def calc_ionhistory(self):
        """ """
        if self.verbose:
            print(f"Calculating ionisation history...")

        z = []
        xe = []
        for slice in self.xion:
            # if self.verbose:
            #    print(f'Now on file {n}')
            if self.verbose:
                print(f"Calculating ionisation fraction at redshift {slice['z']}")

            cube = slice["cube"]
            z.append(slice["z"])
            xe.append(np.mean(cube))

        if self.verbose:
            print("")

        return np.asarray(z), np.asarray(xe)

    def calculate_power(self, cubes, k=None, n_bins=25, log_bins=True, type="the"):
        """

        :param k:  (Default value = None)
        :param n_bins:  (Default value = 25)
        :param log_bins:  (Default value = True)

        """
        P_k = np.zeros((self.z.size, n_bins))
        bins = np.zeros(n_bins)
        for i, z in enumerate(self.z):
            if self.verbose:
                print(f"Calculating {type} power spectrum at redshift {z}")

            field = cubes[i]
            overdensity = (field - np.mean(field)) / np.mean(field)

            if type == "electron":
                ne = self.xion[i] * (1 + overdensity)
                overdensity = (ne - np.mean(ne)) / np.mean(ne)

            pk = 0
            bins = 0
            if k is not None:
                if self.verbose:
                    print("Using the k values you asked for")
                pk, bins = get_power(overdensity, self.box_size, **self.pspec_kwargs)

            if k is None:
                pk, bins = get_power(overdensity, self.box_size, **self.pspec_kwargs)

            P_k[i] = pk

        return P_k, bins
