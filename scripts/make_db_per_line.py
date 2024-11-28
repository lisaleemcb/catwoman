import os
import copy as cp
from string import printable
import logging
import numpy as np
import pandas as pd

import ksz.analyse
import ksz.utils
import ksz.Pee
import catwoman.utils as utils

from scipy.interpolate import CubicSpline, RectBivariateSpline, RegularGridInterpolator
from catwoman.shelter import Cat
from ksz.parameters import *
#from ksz_model import *
#from parameters import *

# logs directory
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

path_spectra = '/obs/emcbride/spectra/Pee'
path_sim = '/obs/emcbride/sims'
Pdd_fn = '/obs/emcbride/kSZ/data/Pk.npz'
errs_fn = '/obs/emcbride/kSZ/data/EMMA/EMMA_frac_errs.npz'
lklhd_path = '/obs/emcbride/lklhd_files'
Pee_path = '/obs/emcbride/Pee_files'
xe_path = '/obs/emcbride/xe_files'
KSZ_path = '/obs/emcbride/KSZ_files'

#Pdd = np.load(Pdd_fn)

k_ref = np.load('/obs/emcbride/catwoman/refs/Pdd/k.npy')
z_ref = np.load('/obs/emcbride/catwoman/refs/Pdd/z.npy')
Pk_ref = np.load('/obs/emcbride/catwoman/refs/Pdd/Pk.npy')

Pdd = {'z': z_ref,
       'k': k_ref,
       'Pk': Pk_ref}

def Pk(k, z, ref=Pdd):
    z = z
    Pdd_shape = (k * z).shape
    #Pdd_interp = np.zeros(Pdd_shape)
    fit_points = [Pdd['z'], Pdd['k']]
    values = np.log10(Pdd['Pk'])

    interp = RegularGridInterpolator(fit_points, values, bounds_error=False, fill_value=np.log10(0.0))
    broadcasted_z = np.broadcast_to(z, Pdd_shape)
    broadcasted_k = np.broadcast_to(k, Pdd_shape)

    interp_params = np.stack([broadcasted_z.flatten(), broadcasted_k.flatten()], axis=1)
    Pdd_interp = 10**interp(interp_params).reshape(Pdd_shape)

    return Pdd_interp

errs = np.load(errs_fn)
EMMA_k = errs['k']
frac_err_EMMA = errs['err']
err_spline  = CubicSpline(EMMA_k, frac_err_EMMA)

# KSZ = KSZ_power(verbose=True)
# Pk =  KSZ.run_camb(force=True, return_Pk=True)

# load simulations that are already parsed so we can skip
skipped = np.load('skipped.npy')
written = np.load('written.npy')

written = written.tolist()
skipped = skipped.tolist()

print('So far the following simulations have been written out:')
print(written)

sims_num = []
print(f'Now parsing simulation directories in {path_sim}')
for dir in os.listdir(path_sim):
    print(f'On sim {dir}')

    basename, extension = os.path.splitext(dir)
    sim, num = basename.split('u')

    sims_num.append(num)

bf_params = {}

for sn in sims_num:
    print('-----------------------------------------------------------')
    print(f'Now on sim {sn}...')

    logger_name = f'logger_{sn}'
    log_file = os.path.join(log_dir, f'simu{sn}.log')
    logger = utils.setup_logger(logger_name, log_file)

    logger.info(f'ON SIMULATION {sn}')
    if sn in written:
        logger.info(f'Already parsed sim {sn}')
        print('Skipping due to already being written!')
        continue

    if sn in skipped:
        logger.info(f'Already parsed sim {sn} and decided to skip')
        print('Skipping due to incomplete sim!')

        continue

    else:
        path_params = '/obs/emcbride/param_files'
        params_file = f'{path_params}/runtime_parameters_simulation_{sn}_reformatted.txt'
        redshift_file = f'{path_sim}/simu{sn}/postprocessing/cubes/lum/redshift_list.dat'

        logger.info(f'params_file={params_file}')
        logger.info(f'redshift_file={redshift_file}')

        if not os.path.isfile(params_file):
            err_file = os.path.join(log_dir, f'simu{sn}.failed')
            logger_err = utils.setup_logger(logger_name, err_file)

            logger_err.warning(f'no params file')
            print('skipping sim {sn}...')
            skipped.append(sn)
            np.save('skipped.npy', skipped)

            continue 

        elif not os.path.isfile(redshift_file) or os.path.getsize(redshift_file) == 0:
            err_file = os.path.join(log_dir, f'simu{sn}.failed')
            logger_err = utils.setup_logger(logger_name, err_file)

            logger_err.warning(f'no redshift file with the extension .dat. Trying .txt...')
            redshift_file = f'{path_sim}/simu{sn}/postprocessing/cubes/lum/redshift_list.txt'
            if not os.path.isfile(redshift_file):
                err_file = os.path.join(log_dir, f'simu{sn}.failed')
                logger_err = utils.setup_logger(logger_name, err_file)

                logger_err.warning(f'no redshift file with the extension .txt')
                print(f'skipping sim {sn}...')
                skipped.append(sn)
                np.save('skipped.npy', skipped)

                continue

        else:
            logger.info('passed preliminary checks...loading sim...')
            sim = Cat(sn,
                        verbose=True,
                        load_params=True,
                        load_xion_cubes=True,
                        load_density_cubes=True,
                        reinitialise_spectra=True,
                        save_spectra=True,
                        just_Pee=True,
                        path_sim=path_sim,
                        path_params=path_params,
                        path_spectra=path_spectra,
                        path_Pee=f'/loreli/rmeriot/ps_ee/simu{sn}/postprocessing/cubes/ps_dtb')

            logger.info(f'xion looks like: {sim.xion}')
            snapshots_file = f'{path_sim}/simu{sn}/snapshots/diagnostics.dat'
            if not os.path.isfile(snapshots_file):
                err_file = os.path.join(log_dir, f'simu{sn}.missingfiles.failed')
                logger_err = utils.setup_logger(logger_name, err_file)
                np.save('skipped.npy', skipped)
                logger_err.warning(f'No snapshot files at {snapshots_file}...skipping sim {sn}')

                continue

            if not sim.density:
                err_file = os.path.join(log_dir, f'simu{sn}.missingfiles.failed')
                logger_err = utils.setup_logger(logger_name, err_file)
                np.save('skipped.npy', skipped)
                logger_err.warning(f'No density cubes at {sim.path_density}...skipping sim {sn} initialisation')

                continue

            if not sim.xion:
                err_file = os.path.join(log_dir, f'simu{sn}.missingfiles.failed')
                logger_err = utils.setup_logger(logger_name, err_file)
                np.save('skipped.npy', skipped)
                logger_err.warning(f'No xion cubes at {sim.path_xion}...skipping sim {sn} initialisation')

                continue


            elif sim.xion:
                if max(sim.xe) < .9:
                    err_file = os.path.join(log_dir, f'simu{sn}.incompletereion.failed')
                    logger_err = utils.setup_logger(logger_name, err_file)
                    skipped.append(sn)
                    np.save('skipped.npy', skipped)
                    logger_err.warning('sim does not complete reionisation')

                    continue

                elif max(sim.xe) >= .9:
                    print('Now onto the science!')
                    logger.info('sim reaches ionisation fraction 90%, starting analysis...')


                    # print('saving files data...')
                    # xe_file = os.path.join(xe_path, f'xe_history_simu{sn}')
                    # np.savez(xe_file, z=sim.z, xe=sim.xe)

                    # Pee_file = os.path.join(Pee_path, f'Pee_simu{sn}')
                    # np.savez(Pee_file, k=sim.k, Pee=sim.Pee, Pbb=sim.Pbb, Pxx=sim.Pxx)

                    #################################
                    #  Fitting for G22 parameters
                    #################################
                    zrange = np.where((sim.xe >= .01) & (sim.xe <= .98))[0]
                    krange = np.where((sim.k >= k_res[0]) & (sim.k <= 2.0))[0]

                    data = sim.Pee[np.ix_(zrange, krange)]
                    obs_errs = err_spline(sim.k[krange]) * data

                    fit = ksz.analyse.Fit(zrange, krange, cp.deepcopy(modelparams_Gorce2022), sim,
                                            data=data, load_errs=False,
                                            initialise=True, Pdd=Pk(sim.k[krange], sim.z[zrange, None]),
                                            debug=False, verbose=False, nsteps=10, obs_errs=obs_errs)
                    
                    # # Combine the directory path and file name to create the full file path using os.path.join
                    lklhd_file = os.path.join(lklhd_path, f'lklhd_simu{sn}')
                    np.save(lklhd_file, fit.lklhd)

                    bf_params[sn] = fit.lklhd_params
                    bf_file = os.path.join(lklhd_path, f'bestfit_params_simu{sn}')

                    print('Saving best fit params...')   
                    np.savez(bf_file, bf=bf_params)

                    # KSZ simulation
 #                   ells = np.linspace(1,12000, 100)
 #                   KSZ = KSZ_power(verbose=True, interpolate_xe=True, interpolate_Pee=True,
 #                           alpha0=fit.lklhd_params['alpha0'], kappa=fit.lklhd_params['kappa'],
 #                           Pee_data=fit.data, xe_data=fit.xe, z_data=fit.z, k_data=fit.k)
   
  #                  KSZ_spectra = KSZ.run_ksz(ells=ells, patchy=True, Dells=True)[:,0]

   #                 KSZ_file = os.path.join(KSZ_path, f'KSZ_simu{sn}')
   #                 np.savez(KSZ_file, ells=ells, KSZ=KSZ_spectra)

                    written.append(sn)
                    np.save('written.npy', written)

                    logger.info(f'Sim {sn} saved to disk...')
                    print(f'Sim {sn} saved to disk...')
                    print('')
            


print('')
print(f'We actually ran through all the files!')
