import matplotlib
import matplotlib.pyplot as plt
import numpy as np

home_dir = "/home/emc-brid"
base_dir = "/data/cluster/emc-brid/Datasets/LoReLi"

box_size = 296.0  # Mpc
k_res = ((2 * np.pi) / box_size, (2 * np.pi * 256) / box_size / 2)
k_bins = np.geomspace(k_res[0], k_res[1], 26)
k_center = []
for i in range(k_bins.size - 1):
    k_center.append((k_bins[i] + k_bins[i + 1]) / 2)

k_center = np.asarray(k_center)

norm_z = matplotlib.colors.Normalize(vmin=3.0, vmax=22)
cmap_z = plt.get_cmap("viridis_r")

norm_xe = matplotlib.colors.LogNorm(vmin=1.7e-4, vmax=1.16)
cmap_xe = plt.get_cmap("plasma")

astro_pnames = ["f_X", "rHS", "tau", "Mmin", "f_esc"]
astro_pnames_formatted = [
    r"$\log{f_X}$",
    r"$r_{\mathrm{H/S}}$",
    r"$\tau_{\mathrm{SF}}$",
    r"$\log{M_{\mathrm{min}}}$",
    r"$f_{\mathrm{esc}}$",
    r"$\tau_{\mathrm{CMB}}$",
    r"$A_{\mathrm{hkSZ}}$",
]

astro_wunits = [
    r"$\log{f_X}$",
    r"$r_{\mathrm{H/S}}$",
    r"$\log{[\tau_{\mathrm{SF}}/\mathrm{Gyr}]}$",
    r"$\log{[M_{\mathrm{min}}/M_{\odot}]}$",
    r"$f_{\mathrm{esc}}$",
    r"$\tau_{\mathrm{CMB}}$",
    r"$A_{\mathrm{bias}}$ ",
    r"$A_{\mathrm{hkSZ}}$",
]
