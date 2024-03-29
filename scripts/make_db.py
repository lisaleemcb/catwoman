import os
import pandas as pd

import catwoman.cat as cat

db_fn = 'Loreli_data.db'

path = '/obs/emcbride/sims'
#path = '/Users/emcbride/kSZ/data/Pee_spectra_LoReLi/formatted/'

sims_num = []
for filename in os.listdir(path):
    basename, extension = os.path.splitext(filename)
    sim, num = basename.split('u')

    sims_num.append(num)

sims = [[] for n in sims_num]
for i, n in enumerate(sims_num):
    sim = cat.Cat(n,
                verbose=True,
                load_params=True,
                path_sim=path)

   # sim.params['spectra'] = sim.spectra
    sims[i] = sim.params


df = pd.DataFrame(sims)
df.to_csv(db_fn)

print('example sim is')
print(sims[5])

print(f'data saved to {db_fn}')
