import os
import numpy as np

import catwoman.cat as cat

sims_num = []
for filename in os.listdir('/Users/emcbride/kSZ/data/Pee_spectra_LoReLi/formatted/'):
    basename, extension = os.path.splitext(filename)
    sim, num = basename.split('u')

    sims_num.append(num)

ion_histories = {}
for sn in sims_num:
    sim = cat.Cat(sn, verbose=False, load_ion=True)
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
