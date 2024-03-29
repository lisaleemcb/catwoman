import os
import pandas as pd
import catwoman.cat as cat

from scipy.interpolate import CubicSpline
from catwoman import utils

path = '/obs/emcbride/sims'
db_fn = 'Loreli_data.db'

# The early redshifts wiggle a bit, so I have to cut out the first few elements to make this a proper function
xe_start = .02
xe_mid = 0.5
xe_end = 0.98

skip = 5 # this is because sometimes xion goes down, which prevents interpolation
baddies = ['10446', '10476', '10500', '10452', '10506'] # sims with crazy ion histories

# sims_num = []
# for filename in os.listdir(path):
#     basename, extension = os.path.splitext(filename)
#     sim, num = basename.split('u')

#     sims_num.append(num)

sims = []
for sn in open("/obs/emcbride/catwoman/refs/sim_nums.txt",'r').read().splitlines():
    if sn in baddies:
        print(f'Skipped the baddie {sn}')

    else:
        print('===================================')
        print(f'Loading sim {sn}')
        print('===================================')

        sim = cat.Cat(sn,
                    verbose=True,
                    load_params=True,
                    load_xion=True,
                    path_sim=path,
                    path_params='/obs/emcbride/param_files')

        z, xe = sim.calc_ion_history()
        spl = CubicSpline(xe[skip:], z[skip:])

        z_start = spl(xe_start)
        z_mid = spl(xe_mid)
        z_end = spl(xe_end)
        A = utils.calc_asymmetry(z_start, z_mid, z_end)
        duration = utils.duration(z_start, z_end)

        sim.params['z_start'] = z_start
        sim.params['z_mid'] = z_mid
        sim.params['z_end'] = z_end
        sim.params['A'] = A
        sim.params['duration'] = duration

        sims.append(sim.params)


df = pd.DataFrame(sims)
df.to_csv(db_fn)

print('example sim is')
print(sims[5])

print(f'data saved to {db_fn}')
