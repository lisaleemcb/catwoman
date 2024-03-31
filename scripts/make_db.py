import os
import numpy as np
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

#skip = 5 # this is because sometimes xion goes down, which prevents interpolation
baddies = ['10446', '10476', '10500', '10452', '10506'] # sims with crazy ion histories
empties = ['17941', '10598', '19086', '12022', '10468', '17989', '13219', '10197', '16691', '10692', '19085'] # doesn't contain a critical file

sims_num = []
for dir in os.listdir(path):
    print(f'Now parsing simulation directories in {path}')
    print(f'On sim {dir}')

    basename, extension = os.path.splitext(dir)
    sim, num = basename.split('u')

    sims_num.append(num)

sims = []
# for sn in open("/obs/emcbride/catwoman/refs/sim_nums.txt",'r').read().splitlines():
for sn in sims_num:
    if sn in baddies:
        print(f'Skipped the baddie {sn}')
    if sn in empties:
        print(f'Skipped empty {sn}')

    else:
        print('===================================')
        print(f'Loading sim {sn}')
        print('===================================')

        path_params = '/obs/emcbride/param_files'
        params_file = f'{path_params}/runtime_parameters_simulation_{self.sim_n}_reformatted.txt'
        if not os.path.isfile(params_file):
            print(f'Skipped sim {sn}, added empty sim to list')
            empties.append(sn)

        sim = cat.Cat(sn,
                    verbose=True,
                    load_params=True,
                    load_xion=False,
                    path_sim=path,
                    path_params=path_params)

        #z, xe = sim.calc_ion_history()

        snapshots_file = f'{path}/simu{sn}/snapshots/diagnostics.dat'
        if not os.path.isfile(snapshots_file):
            print(f'Skipped sim {sn}, added empty sim to list')
            empties.append(sn)
        else:
            snapshots = np.genfromtxt(snapshots_file)
            z = snapshots[:,0]
            xe = snapshots[:,1]

            skip = utils.find_index(xe)

            # interpolation to get z(xe)
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

print('saving empties list...')
np.save('empties', empties)

print('example sim is')
print(sims[5])

print(f'data saved to {db_fn}')
