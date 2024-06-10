import os
import numpy as np
import pandas as pd

import ksz.analyse
import ksz.utils
import ksz.Pee

from scipy.interpolate import CubicSpline
from catwoman import utils, catwoman
from ksz.parameters import *

path = '/obs/emcbride/sims'
Pdd_fn = '/obs/emcbride/kSZ/data/Pdd.npy'
errs_fn = '/obs/emcbride/kSZ/data/EMMA/EMMA_frac_errs.npz'
lklhd_path = '/obs/emcbride/lklhd_files'
spectra_path = '/obs/emcbride/spectra_files'
xe_path = '/obs/emcbride/xe_files'

Pdd = np.load(Pdd_fn)
errs = np.load(errs_fn)
EMMA_k = errs['k']
frac_err_EMMA = errs['err']
err_spline  = CubicSpline(EMMA_k, frac_err_EMMA)

# sims with crazy ion histories
baddies = ['10446', '10476', '10500', '10452', '10506']

# load simulations that are already parsed so we can skip
written = np.load('written.npy')
written = written.tolist()

print('So far the following simulations have been written out:')
print(written)

sims_num = []
for dir in os.listdir(path):
    print(f'Now parsing simulation directories in {path}')
    print(f'On sim {dir}')

    basename, extension = os.path.splitext(dir)
    sim, num = basename.split('u')

    sims_num.append(num)


for sn in sims_num:
    print('-----------------------------------------------------------')
    if sn in written:
        print(f'Already parsed sim {sn}')
    elif sn in baddies:
        print(f'Skipped the baddie {sn}')
    else:
        path_params = '/obs/emcbride/param_files'
        params_file = f'{path_params}/runtime_parameters_simulation_{sn}_reformatted.txt'
        redshift_file = f'{path}/simu{sn}/postprocessing/cubes/lum/redshift_list.dat'

        if not os.path.isfile(params_file):
            print(f'Skipped sim {sn}')
        elif not os.path.isfile(redshift_file):
            print(f'Skipped sim {sn}')
        else:
            sim = catwoman.Cat(sn,
                        verbose=True,
                        load_params=True,
                        load_xion=True,
                        load_density=True,
                        initialise_spectra=True,
                        path_sim=path,
                        path_params=path_params,
                        path_Pee=f'/loreli/rmeriot/ps_ee/simu{sn}/postprocessing/cubes/ps_dtb')

            snapshots_file = f'{path}/simu{sn}/snapshots/diagnostics.dat'
            if not os.path.isfile(snapshots_file):
                print(f'Skipped sim {sn}')
            if not sim.xion:
                print(f'Skipping sim {sn} initialisation due to missing files')
            elif sim.xion:
                if max(sim.xe) > .9:
                    print('Now onto the science!')

                    #################################
                    #  Fitting for G22 parameters
                    #################################
                    k0 = 3
                    kf = 18
                    krange = (k0, kf)

                    z0 = np.where(sim.xe > .01)[0][0]
                    zf = np.where(sim.xe > .9)[0][0] + 1
                    zrange = (z0, zf)

                    z_inter = np.linspace(5,25, 100)
                    Pdd_spline = CubicSpline(z_inter, Pdd[:,k0:kf])
                    Pdd_inter = Pdd_spline(sim.z[z0:zf])

                    truths = [np.log10(modelparams_Gorce2022['alpha_0']), modelparams_Gorce2022['kappa']]
                    priors =[(np.log10(modelparams_Gorce2022['alpha_0']) * .25, np.log10(modelparams_Gorce2022['alpha_0']) * 1.75),
                            (0, modelparams_Gorce2022['kappa'] * 5.0),
                            (modelparams_Gorce2022['k_f'] * .25, modelparams_Gorce2022['k_f'] * 5.0),
                            (modelparams_Gorce2022['g'] * .25, modelparams_Gorce2022['g'] * 5.0)]

                    fit2 = ksz.analyse.Fit(zrange, krange, modelparams_Gorce2022, sim, priors,
                                                      frac_err=err_spline(sim.k[k0:kf]),
                                                      Pdd=Pdd_inter, ndim=2, initialise=False)

                    a0 = np.linspace(*priors[0], num=100)
                    kappa = np.linspace(*priors[1], num=120)
                    lklhd_grid = np.zeros((a0.size, kappa.size))

                    for i, ai in enumerate(a0):
                        for j, ki in enumerate(kappa):
                            lklhd_grid[i,j] = ksz.analyse.log_like((ai, ki), fit2.data, fit2.model_func,
                                                                        priors, fit2.obs_errs)


                    # Combine the directory path and file name to create the full file path using os.path.join
                    lklhd_file = os.path.join(lklhd_path, f'lklhd_grid_simu{sn}')
                    np.save(lklhd_file, lklhd_grid)

                    xe_file = os.path.join(xe_path, f'xe_history_simu{sn}')
                    np.savez(xe_file, z=sim.z, xe=sim.xe)

                    spectra_file = os.path.join(spectra_path, f'spectra_simu{sn}')
                    np.savez(spectra_file, Pee=sim.Pee, Pbb=sim.Pbb)

                    written.append(sn)
                    np.save('written.npy', written)

                    print(f'Sim {sn} saved to disk...')

print(f'We actually ran through all the files!')
