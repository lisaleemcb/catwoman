import os
import numpy as np

import catwoman.cat as cat



# sims_num = []
# for filename in os.listdir('/Users/emcbride/kSZ/data/Pee_spectra_LoReLi/formatted/'):
#     basename, extension = os.path.splitext(filename)
#     sim, num = basename.split('u')

#     sims_num.append(num)

ion_histories = {}
for sn in open("/obs/emcbride/catwoman/refs/sim_nums.txt",'r').read().splitlines():
    sim = cat.Cat(sn,
                    verbose=False,
                    load_ion=True,
                    load_spectra=True,
                    path_sim='/loreli/rmeriot',
                    path_params = 'Pee_spectra_LoReLi/formatted',
                    path_Pee = 'ps_ee',
                    path_ion = '/obs/emcbride/xion',
                    path_dens = 'dens')
    z = []
    XII = []
    for i, n in enumerate(sim.file_nums):
        if sim.spectra[i]['file_n'] == n:
            z.append(sim.spectra[i]['z'])
            XII.append(np.mean(sim.ion_cube[i]['ion_cube']))

    history = {'z': z,
               'XII': XII}
    ion_histories[sn] = history

np.savez('ion_histories', ion_histories)
