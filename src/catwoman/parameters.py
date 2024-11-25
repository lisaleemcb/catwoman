import numpy as np
from astropy import cosmology, units, constants


######################################
########### FIT PARAMETERS ###########
######################################

blobnames = [
    "pksz",
    "hksz",
    "tt_cmb",
    "ee_cmb",
    "te_cmb"
    "tau",
]
# priors
z_max = 20.0
xe_recomb = 1.7e-4

#####################################
############# COSMOLOGY #############
#####################################

T_cmb = 2.7255
Yp = 0.2453

#######################################
########### System settings ###########
#######################################

Mpcm = (1.0 * units.Mpc).to(units.m).value  # one Mpc in [m]
Mpckm = Mpcm / 1e3

#######################################
###### REIONISATION PARAMETERS ########
#######################################

# reionisation of Helium
helium_fullreion_redshift = 3.5
helium_fullreion_start = 5.0
helium_fullreion_deltaredshift = 0.5

########################################
#### Integration/precision settings ####
########################################

# minimal and maximal values valid for CAMB interpolation of
kmax_pk = 1570.5

### Settings for theta integration
num_th = 50
th_integ = np.linspace(0.0001, np.pi * 0.9999, num_th)
mu = np.cos(th_integ)  # cos(k.k')

### Settings for k' (=kp) integration
# k' array in [Mpc-1] - over which you integrate
min_logkp = -5.0
max_logkp = 1.5
dlogkp = 0.05
kp_integ = np.logspace(min_logkp, max_logkp, int((max_logkp - min_logkp) / dlogkp) + 1)
# minimal and maximal values valid for CAMB interpolation of
kmax_camb = 6.0e3
kmin_camb = 7.2e-6
krange_camb = np.logspace(np.log10(kmin_camb), np.log10(kmax_camb), 500)

### Settings for z integration
z_recomb = 1100.0
z_min = 0.10
z_piv = 1.0
z_max = 20.0
dlogz = 0.1
dz = 0.15
z_integ = np.concatenate(
    (
        np.logspace(
            np.log10(z_min),
            np.log10(z_piv),
            int((np.log10(z_piv) - np.log10(z_min)) / dlogz) + 1,
        ),
        np.arange(z_piv + dz, 10.0, step=dz),
        np.arange(10, z_max + 0.5, step=0.5),
    )
)
z3 = np.linspace(0, z_recomb, 10000)


###################################
###### ANALYSIS PARAMETERS ########
###################################

# statistical parameters
CL = 95  # confidence interval
percentile1 = (100 - CL) / 2
percentile2 = CL + (100 - CL) / 2
smoothing = 1.0

# plotting parameters
ylabels = [ 'TT', 'EE', 'TE', 'pkSZ', 'hkSZ']
props = dict(boxstyle="round", facecolor="white", alpha=0.5)
colorlist = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]
cmaps = ["Blues", "Oranges", "Greens", "PuRd"]
alphas = [0.5, 0.5, 0.5, 0.9]
smooth = 1.0
