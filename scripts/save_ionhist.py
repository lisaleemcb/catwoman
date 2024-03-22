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

path = '/obs/emcbride/sims'

baddies = ['17681', '13492']
sims_num = []
sims_none = []
for filename in os.listdir(path):
    basename, extension = os.path.splitext(filename)
    sim, num = basename.split('u')

    sims_num.append(num)

for i, sn in enumerate(sims_num):
    if sn in baddies:
        print(f'Skipped the baddie {sn}')

    else:
        print('===================================')
        print(f'Loading sim {sn}')
        print('===================================')
        sim = cat.Cat(sn,
                        verbose=True,
                        load_params=False,
                        load_Pee=False,
                        load_ion=True,
                        load_density=True,
                        path_sim=path)
        print(f'files: {sim.file_nums}')
        print(f'with redshifts: {sim.redshifts}')

        z = []
        xe = []

        if len(sim.file_nums) == 0:
            print('No ion files in this sim :(')
            sims_none.append(i)
        if len(sim.file_nums) > 0:
            sims_num.append(i)
            for j, fn in enumerate(sim.file_nums):
                print('Now calculating the ion history...')
                if sim.ion[j]['file_n'] == fn:
                    z.append(sim.ion[j]['z'])
                    xe.append(np.mean(sim.ion[j]['cube']))

        history = {'z': z,
                'xe': xe}
        ion_histories[sn] = history

        print('===================================')

np.save('sims_num.txt', sims_num)
np.save('sims_none.txt', sims_none)
np.savez('ion_histories', ion_histories)
