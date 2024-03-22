import os
import numpy as np

import catwoman.cat as cat


ion_histories = {}
# for sn in open("/obs/emcbride/catwoman/refs/sim_nums.txt",'r').read().splitlines():
#     sim = cat.Cat(sn,
#                     verbose=True,
#                     load_ion=True,
#                     load_density=True,
#                     load_Pee=False,
#                     path_sim='/obs/emcbride/sims',
#                     path_params = 'Pee_spectra_LoReLi/formatted',
#                     path_Pee = 'ps_ee',
#                     path_ion = '/obs/emcbride/xion',
#                     path_dens = 'dens')
#
sims_num = []
for filename in os.listdir(path):
    basename, extension = os.path.splitext(filename)
    sim, num = basename.split('u')

    sims_num.append(num)

sims = [[] for n in sims_num]
for i, n in enumerate(sims_num):
    sim = cat.Cat(n,
                verbose=False,
                load_params=False,
                load_spectra=False)
   # sim.params['spectra'] = sim.spectra
   #
    z = []
    xe = []
    for i, n in enumerate(sim.file_nums):
        if sim.spectra[i]['file_n'] == n:
            z.append(sim.spectra[i]['z'])
            xe.append(np.mean(sim.ion_cube[i]['ion_cube']))

    history = {'z': z,
               'xe': xe}
    ion_histories[sn] = history

np.savez('ion_histories', ion_histories)
