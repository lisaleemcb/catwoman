import os
import numpy as np
import pandas as pd
import catwoman.cat as cat

from scipy.interpolate import CubicSpline
from catwoman import utils

path = '/obs/emcbride/sims'
db_fn = 'Loreli_data.db'
ion_fn = 'ion_histories'
empties_fn = 'empties'
Pee_spectra_fn = 'Pee_spectra'

# The early redshifts wiggle a bit, so I have to cut out the first few elements to make this a proper function
xe_start = .02
xe_mid = 0.5
xe_end = 0.98

#skip = 5 # this is because sometimes xion goes down, which prevents interpolation
baddies = ['10446', '10476', '10500', '10452', '10506'] # sims with crazy ion histories
empties = [] # doesn't contain a critical file

sims_num = []
for dir in os.listdir(path):
    print(f'Now parsing simulation directories in {path}')
    print(f'On sim {dir}')

    basename, extension = os.path.splitext(dir)
    sim, num = basename.split('u')

    sims_num.append(num)

ion_histories = {}
Pee_spectra = {}
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
        params_file = f'{path_params}/runtime_parameters_simulation_{sn}_reformatted.txt'
        redshift_file = f'{path}/simu{sn}/postprocessing/cubes/lum/redshift_list.dat'

        if not os.path.isfile(params_file):
            print(f'Skipped sim {sn}, added empty sim to list')
            empties.append(sn)
        elif not os.path.isfile(redshift_file):
            print(f'Skipped sim {sn}, added empty sim to list')
            empties.append(sn)

        else:
            sim = cat.Cat(sn,
                        verbose=True,
                        load_params=True,
                        load_xion=True,
                        load_density=True,
                        path_sim=path,
                        path_params=path_params,
                        path_Pee=f'/loreli/rmeriot/ps_ee/simu{sn}/postprocessing/cubes/ps_dtb')

            #z, xe = sim.calc_ion_history()

            snapshots_file = f'{path}/simu{sn}/snapshots/diagnostics.dat'
            if not os.path.isfile(snapshots_file):
                print(f'Skipped sim {sn}, added empty sim to list')
                empties.append(sn)
            else:
                print('Now onto the science!')
                snapshots = np.genfromtxt(snapshots_file)
                z = snapshots[:,0]
                xe = snapshots[:,1]

                skip = utils.find_index(xe)

                history = {'z': z[skip:],
                        'xe': xe[skip:]}
                ion_histories[sn] = history

                Pee = sim.calc_Pee()
                Pee_spectra[sn] = Pee

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

                print('Summary statistics saved to params dict...')


print(f'saving database to {db_fn}...')
df = pd.DataFrame(sims)
df.to_csv(db_fn)

print(f'saving empties list to {empties_fn}...')
np.save(empties_fn, empties)

print(f'saving ionisation histories to {ion_fn}...')
np.savez(ion_fn, ion_histories)

print(f'saving electron power spectra to {Pee_spectra_fn}...')
np.savez(Pee_spectra_fn, Pee_spectra)

print('example sim is')
print(sims[5])

print(f'done!')
