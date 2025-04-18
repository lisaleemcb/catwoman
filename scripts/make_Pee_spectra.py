import os
import copy as cp
from string import printable
import logging
import numpy as np
import pandas as pd

import catwoman.utils as utils
from catwoman.shelter import Cat
#from ksz.parameters import *


import argparse

# logs directory
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

parser = argparse.ArgumentParser(description="Load a numpy file which is a list of sims and save to a directory.")
parser.add_argument("--sims", type=str, help="Path to the numpy file (.npy or .npz) with the sims to parse")
args = parser.parse_args()

redshifts_fn = 'obs/emcbride/redshift_list.dat'
path_spectra = '/obs/emcbride/spectra/Pee'
path_sim = '/loreli/rmeriot/simus_loreli/' #'/obs/emcbride/sims'  # head folder holding all the simulation cubes

# load simulations that are already parsed so we can skip
# skipped = np.load('skipped.npy')
# written = np.load('written.npy')

# written = written.tolist()
# skipped = skipped.tolist()

skipped = []
written = []

print('So far the following simulations have been written out:')
print(written)

if args.sims is not None:
    sims_num = np.load(args.sims, allow_pickle=True)

else:
    sims_num = []
    print(f'Now parsing simulation directories in {path_sim}')
    for dir in os.listdir(path_sim):
        print(f'On sim {dir}')

        basename, extension = os.path.splitext(dir)
        sim, num = basename.split('u')

        sims_num.append(num)

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
        print('Skipping since in skipping file!')

        continue

    else:
        path_params = '/obs/emcbride/param_files'
        params_file = f'{path_params}/runtime_parameters_simulation_{sn}_reformatted.txt'

        logger.info(f'params_file={params_file}')
        logger.info(f'redshift_file={redshifts_fn}')

        if not os.path.isfile(params_file):
            err_file = os.path.join(log_dir, f'simu{sn}.failed')
            logger_err = utils.setup_logger(logger_name, err_file)

            logger_err.warning(f'no params file')
            print('skipping sim {sn}...')
            skipped.append(sn)
            np.save('skipped.npy', skipped)

            continue 

        else:
            logger.info('passed preliminary checks...loading sim...')
            sim = Cat(sn,
                        verbose=True,
                        load_params=False,
                        skip_early=False,
                        load_spectra=False,
                        reinitialise_spectra=True,
                        save_spectra=True,
                        just_Pee=True,
                        base_dir=None,
                        redshifts_fn=redshifts_fn,
                        path_sim=path_sim,
                        path_params=path_params,
                        path_spectra=path_spectra)

            logger.info(f'xion looks like: {sim.xion}')
            snapshots_file = f'{path_sim}/simu{sn}/snapshots/diagnostics.dat'
            if not os.path.isfile(snapshots_file):
                err_file = os.path.join(log_dir, f'simu{sn}.missingfiles.failed')
                logger_err = utils.setup_logger(logger_name, err_file)
                skipped.append(sn)
                np.save('skipped.npy', skipped)
                logger_err.warning(f'No snapshot files at {snapshots_file}...skipping sim {sn}')

                continue

            if not sim.density:
                err_file = os.path.join(log_dir, f'simu{sn}.missingfiles.failed')
                logger_err = utils.setup_logger(logger_name, err_file)
                skipped.append(sn)
                np.save('skipped.npy', skipped)
                logger_err.warning(f'No density cubes at {sim.path_density}...skipping sim {sn} initialisation')

                continue

            if not sim.xion:
                err_file = os.path.join(log_dir, f'simu{sn}.missingfiles.failed')
                logger_err = utils.setup_logger(logger_name, err_file)
                skipped.append(sn)
                np.save('skipped.npy', skipped)
                logger_err.warning(f'No xion cubes at {sim.path_xion}...skipping sim {sn} initialisation')

                continue

            # elif sim.xion:
            #     if max(sim.xe) < .9:
            #         err_file = os.path.join(log_dir, f'simu{sn}.incompletereion.failed')
            #         logger_err = utils.setup_logger(logger_name, err_file)
            #         skipped.append(sn)
            #         np.save('skipped.npy', skipped)
            #         logger_err.warning('sim does not complete reionisation')

            #         continue

                # elif(np.isnan(utils.find_index(sim.xe))):
                #     err_file = os.path.join(log_dir, f'simu{sn}.incompletereion.failed')
                #     logger_err = utils.setup_logger(logger_name, err_file)
                #     skipped.append(sn)
                #     np.save('skipped.npy', skipped)
                #     logger_err.warning('sim does not complete reionisation')

                #     continue

            written.append(sn)
            np.save('written.npy', written)

            logger.info(f'Sim {sn} saved to disk...')
            print(f'Sim {sn} saved to disk...')
            print('')
            


print('')
print(f'We actually ran through all the files!')
