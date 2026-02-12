# from ksz.parameters import *
import argparse
import copy as cp
import logging
import os
from string import printable

import numpy as np
import pandas as pd

import catwoman.utils as utils
from catwoman.shelter import Cat

parser = argparse.ArgumentParser(
    description="Load a numpy file which is a list of sims and save to a directory."
)
parser.add_argument(
    "--sims",
    type=str,
    help="Path to the numpy file (.npy or .npz) with the sims to parse",
)

parser.add_argument(
    "--redshift_file",
    type=str,
    help="File number associated with desired redshift",  # see redshift_keys in utils.py for mapping
)

parser.add_argument(
    "--field",
    type=str,
    required=True,
    help="Which field to slice",  # see redshift_keys in utils.py for mapping
)

args = parser.parse_args()

home_dir = "/obs/emcbride"

# logs directory
log_dir = f"logs_{args.field}"
os.makedirs(log_dir, exist_ok=True)

if args.field == "density":
    path_slices = f"{home_dir}/density_slices"
elif args.field == "T21cm":
    path_slices = f"{home_dir}/T21cm_slices"

path_sim = "/loreli/rmeriot/simus_loreli"  #'/obs/emcbride/sims'  # head folder holding all the simulation cubes

# # load simulations that are already parsed so we can skip
# skipped = np.load(f'{home_dir}/skipped.npy')
# written = np.load(f'{home_dir}/written.npy')

# written = written.tolist()
# skipped = skipped.tolist()

if args.sims is not None:
    sims_num = np.load(args.sims, allow_pickle=True)

else:
    sims_num = []
    print(f"Now parsing simulation directories in {path_sim}")
    for dir in os.listdir(path_sim):
        print(f"On sim {dir}")

        basename, extension = os.path.splitext(dir)
        sim, num = basename.split("u")

        sims_num.append(num)

if args.redshift_file is not None:
    filenum = args.redshift_file
else:
    filenum = "032"  # Corresponding to a redshift of 7.431433

z = utils.redshift_keys[filenum]

for sn in sims_num:
    print("-----------------------------------------------------------")
    print(f"Now on sim {sn}...")

    if args.field == "density":
        save_fn = f"{path_slices}/sim{sn}_densityslices_z{z:.2f}"
    elif args.field == "T21cm":
        save_fn = f"{path_slices}/sim{sn}_T21cmslices_z{z:.2f}"

    if os.path.exists(save_fn):
        print("File already exists")
        continue

    logger_name = f"logger_{sn}"
    log_file = os.path.join(log_dir, f"simu{sn}.log")
    logger = utils.setup_logger(logger_name, log_file)

    logger.info(f"ON SIMULATION {sn}")

    if args.field == "density":
        filename = (
            f"{path_sim}/simu{sn}/postprocessing/cubes/dens/dens_256_out{filenum}.dat"
        )
    elif args.field == "T21cm":
        filename = f"{path_sim}/simu{sn}/postprocessing/cubes/dtb/dtb_tp_hi_256_nocorrection_out{filenum}.dat"

    cube = utils.read_cube(filename)

    if cube == 0:
        logger.info(f"Sim {sn} file has a problem...")
        continue

    cube = utils.convert_density(cube, utils.redshift_keys[filenum])

    slices = np.stack([cube[0, :, :], cube[:, 0, :], cube[:, :, 0]])
    np.save(save_fn, slices)

    logger.info(f"Sim {sn} saved to disk...")
    print(f"Sim {sn} saved to disk...")
    print("")

    handlers = logger.handlers[:]
    for h in handlers:
        h.close()
        logger.removeHandler(h)

print("")
print(f"We actually ran through all the files!")
