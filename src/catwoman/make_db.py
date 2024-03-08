import os
import pandas as pd

import cat

sims_num = []
for filename in os.listdir('/Users/emcbride/kSZ/data/Pee_spectra_LoReLi/formatted/'):
    basename, extension = os.path.splitext(filename)
    sim, num = basename.split('u')

    sims_num.append(num)

sims = [[] for n in sims_num]
for i, n in enumerate(sims_num):
    sim = cat.Cat(n, verbose=False)
    # df = pd.DataFrame.from_dict(sim.params, orient='index').T
    # df.info(verbose=False, memory_usage="deep")
    sims[i] = sim.params

print('example sim is')
print(sims[5])
