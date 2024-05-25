import os
import numpy as np
import pandas as pd

import ksz.analyse
import ksz.utils
import ksz.Pee

from scipy.interpolate import CubicSpline
from catwoman import utils, catwoman
from ksz.parameters import *

path = '/obs/emcbride/sims'
Pdd_fn = '/obs/emcbride/kSZ/data/Pdd.npy'
Pdd = np.load(Pdd_fn)

print('Pdd is loaded with shape', Pdd.shape)

db_fn = 'Loreli_data.db'
ion_fn = 'ion_histories'
empties_fn = 'empties'
Pee_spectra_fn = 'Pee_spectra'
fits2_fn = '2paramfit'
fits4_fn = '4paramfit'

# The early redshifts wiggle a bit, so I have to cut out the first few elements to make this a proper function
xe_start = .02
xe_mid = 0.5
xe_end = 0.98

#skip = 5 # this is because sometimes xion goes down, which prevents interpolation
baddies = ['10446', '10476', '10500', '10452', '10506', '13321', '13356'] # sims with crazy ion histories
empties = [] # doesn't contain a critical file

sims_num = []
for dir in os.listdir(path):
    print(f'Now parsing simulation directories in {path}')
    print(f'On sim {dir}')

    basename, extension = os.path.splitext(dir)
    sim, num = basename.split('u')

    sims_num.append(num)

ion_histories = {}
tensions = {}
fits2 = {}
fits4 = {}
Pee_spectra = {}
Pbb_spectra = {}
sims = []

# for sn in open("/obs/emcbride/catwoman/refs/sim_nums.txt",'r').read().splitlines():
for sn in sims_num:
    if sn in baddies:
        print(f'Skipped the baddie {sn}')
    elif sn in empties:
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
            sim = catwoman.Cat(sn,
                        verbose=True,
                        load_params=True,
                        load_xion=True,
                        load_density=True,
                        initialise_spectra=True,
                        path_sim=path,
                        path_params=path_params,
                        path_Pee=f'/loreli/rmeriot/ps_ee/simu{sn}/postprocessing/cubes/ps_dtb')

            snapshots_file = f'{path}/simu{sn}/snapshots/diagnostics.dat'
            if not os.path.isfile(snapshots_file):
                print(f'Skipped sim {sn}, added empty sim to list')
                empties.append(sn)
            if not sim.xion:
                print(f'Skipping sim {sn} initialisation due to missing files')
            else:
                print('Now onto the science!')
                snapshots = np.genfromtxt(snapshots_file)
                z = snapshots[:,0]
                xe = snapshots[:,1]

                skip = utils.find_index(xe)

                #################################
                #  Ionisation histories
                #################################

                history = {'z': z[skip:],
                        'xe': xe[skip:]}
                ion_histories[sn] = history

                #################################
                #  "Tension" parameters
                #################################

                # tension = utils.tension(sim)
                # tensions[sn] = tension

                #################################
                #  Fitting for G22 parameters
                #################################
                k0 = 3
                kf = 18
                krange = (k0, kf)

                z0 = np.where(sim.xe > .01)[0][0]
                zf = np.where(sim.xe > .9)[0][0] + 1
                zrange = (z0, zf)

                z_inter = np.linspace(5,25, 100)
                Pdd_spline = CubicSpline(z_inter, Pdd[:,k0:kf])
                Pdd_inter = Pdd_spline(sim.z[z0:zf])

                truths = [np.log10(modelparams_Gorce2022['alpha_0']), modelparams_Gorce2022['kappa']]
                priors =[(np.log10(modelparams_Gorce2022['alpha_0']) * .25, np.log10(modelparams_Gorce2022['alpha_0']) * 1.75),
                         (0, modelparams_Gorce2022['kappa'] * 5.0),
                         (modelparams_Gorce2022['k_f'] * .25, modelparams_Gorce2022['k_f'] * 5.0),
                         (modelparams_Gorce2022['g'] * .25, modelparams_Gorce2022['g'] * 5.0)]

                fit2 = ksz.analyse.Fit(zrange, krange, modelparams_Gorce2022, sim, priors, Pdd=Pdd_inter, ndim=2)
                fit4 = ksz.analyse.Fit(zrange, krange, modelparams_Gorce2022, sim, priors, Pdd=Pdd_inter, ndim=4, burnin=1000, nsteps=int(1e5))

                fits2[sn] = fit2
                fits4[sn] = fit4

                #################################
                #  Spectra
                #################################
                Pee_spectra[sn] = sim.Pee
                Pbb_spectra[sn] = sim.Pbb

                #################################
                #  Reionisation statistics
                #################################
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
               # sim.params['z_tension'] =
               #
                sim.params['alpha_0'] = fit4.fit_params['alpha_0']
                sim.params['kappa'] = fit4.fit_params['kappa']
                sim.params['a_xe'] = fit4.fit_params['a_xe']
                sim.params['k_xe'] = fit4.fit_params['k_xe']

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

print(f'saving 2-parameter fit to {fits2_fn}...')
np.savez(fits2_fn, fit2)

print(f'saving 4-parameter fit to {fits4_fn}...')
np.savez(fits4_fn, fit4)

print('example sim is')
print(sims[5])

print(f'done!')
