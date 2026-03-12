#! /usr/bin/env python3
"""
Discretizing the hybridization function into a bath
"""

def direct_discretization(w, rho, bins):
    """
    Integrals of each bin

    Returns
    -------
    energies: bath energies :math:`\epsilon_i`
    weights: square of the coupling coefficients :math:`\vert V_i \vert^2`
    """
    dw = (w[-1] - w[0]) / w.size
    e = np.zeros((bins.size-1))
    v2 = np.zeros((bins.size-1))
    for i in range(bins.size-1):
        idx = (np.argwhere((w >= bins[i]) & (w <= bins[i+1])).T)[0]
        #print("Indices = ", idx)
        v2[i] = np.sum(rho[idx]) * dw
        e[i] = np.sum(w[idx] * rho[idx]) * dw / v2[i]
    return e, v2

def gaussian_broadening(w, e, v, eta):
    """
    Spectrum with Gaussian broadening (delta peaks -> Gaussian function)
    of sdtdev eta
    """
    spect = np.zeros_like(w)
    for ei, vi in zip(e, v):
        spect += vi * np.exp(-(w - ei)**2 / (2*eta**2))
    return spect  / eta / np.sqrt(2*np.pi)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import h5py

with h5py.File("./hybridization.h5", 'r') as f:
    print(f.keys())
    it = 11
    print(f["iteration {}".format(it)].keys())
    err = 1e-1
    w = np.array(f["iteration {}/error {}/frequency".format(it,err)])
    hyb = np.array(f["iteration {}/error {}/hybridization".format(it,err)])

full_hyb_eg =(hyb[:,0,0] + hyb[:,1,1]) /2
full_hyb_t2g = (hyb[:,2,2] + hyb[:,3,3] + hyb[:,4,4]) /3
full_w_mesh = w

# below Ef
wmin = -20
wmax = 20.
idx_subEf = np.argwhere((w>wmin) & (w < wmax)).flatten()
print(idx_subEf)
w_mesh = full_w_mesh[idx_subEf]
hyb_eg = full_hyb_eg[idx_subEf]
hyb_t2g = full_hyb_t2g[idx_subEf]

Nbins = 5
bins = np.linspace(w_mesh[0], w_mesh[-1], Nbins)
print(bins)
# discard small coupling terms
threshold = 1e-2
# gaussian broadening
eta = (w_mesh[-1] - w_mesh[0]) / Nbins /2
print("eta = ", eta)

# plot
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1

plot_horizontal = True
if plot_horizontal:
    #fig, ax = plt.subplots(1,1, figsize=(5,3))
    fig, ax = plt.subplots(1,1, figsize=(3.2, 2))
    ax.spines[['right', 'top']].set_visible(False)

    ax.scatter(bins, np.zeros_like(bins), marker='+', c='k', label='bins', zorder=3)
    ax.fill_between(full_w_mesh, -full_hyb_eg.imag/np.pi, color='dodgerblue', alpha=0.3)
    ax.fill_between(full_w_mesh, -full_hyb_t2g.imag/np.pi, color='orangered', alpha=0.3)

    ax.set_xlabel("$\omega$ [eV]")
    ax.set_ylabel("$\Delta(\omega)$, $|V_b|$ [eV]")

else:
    fig, ax = plt.subplots(1,1, figsize=(2.5,3))
    ax.spines[['right', 'top']].set_visible(False)

    ax.scatter(np.zeros_like(bins), bins, marker='+', c='k', label='bins', zorder=3)
    ax.fill_betweenx(full_w_mesh, -full_hyb_eg.imag/np.pi, color='dodgerblue', alpha=0.3)
    ax.fill_betweenx(full_w_mesh, -full_hyb_t2g.imag/np.pi, color='orangered', alpha=0.3)

    ax.set_xlabel("$\Delta(\omega)$, $|V_b|$ [eV]")
    ax.set_yticks(np.linspace(wmin, wmax, 5))
    ax.set_ylabel("$\omega$ [eV]", loc='top', labelpad=-90., rotation='horizontal')

### Eg
rho = - hyb_eg.imag / np.pi
e, v2 = direct_discretization(w_mesh, rho, bins)
v = np.sqrt(v2)
# select the nonzero baths
idx = (np.argwhere(v > threshold).T)[0]
baths = np.stack((e[idx], v[idx])).T
Nbath_Eg = baths.shape[0]
print("\n[Eg] Nonzero baths: ", baths.shape[0])
print("\t {:^20} \t {:^20}".format("epsilon_i", "V_i"))
for i in range(baths.shape[0]):
    print("\t {:^20} \t {:^20}".format(baths[i,0], baths[i,1]))
# broaden
rho_gauss = gaussian_broadening(w_mesh, e[idx], v2[idx], eta)
### plot
if plot_horizontal:
    ax.bar(e[idx], v[idx], width=0.4, color='dodgerblue')
    #ax.plot(w_mesh, rho_gauss, lw=1, c='dodgerblue')
else:
    ax.barh(e[idx], width=v[idx], height=0.4, color='dodgerblue')
    #ax.plot(w_mesh, rho_gauss, lw=1, c='dodgerblue')


### T2g
rho = -hyb_t2g.imag / np.pi
e, v2 = direct_discretization(w_mesh, rho, bins)
v = np.sqrt(v2)
# select the nonzero baths
idx = (np.argwhere(v > threshold).T)[0]
baths = np.stack((e[idx], v[idx])).T
Nbath_T2g = baths.shape[0]
print("\n[T2g] Nonzero baths: ", baths.shape[0])
print("\t {:^20} \t {:^20}".format("epsilon_i", "V_i"))
for i in range(baths.shape[0]):
    print("\t {:^20} \t {:^20}".format(baths[i,0], baths[i,1]))
# broaden
rho_gauss = gaussian_broadening(full_w_mesh, e[idx], v2[idx], eta)
### plot
if plot_horizontal:
    ax.bar(e[idx], v[idx], width=0.4, color='orangered')
    #ax.plot(full_w_mesh, rho_gauss, lw=1, c='orangered')
else:
    ax.barh(e[idx], width=v[idx], height=0.4, color='orangered')
    #ax.plot(w_mesh, rho_gauss, lw=1, c='orangered')

custom_leg1 = [Line2D([0],[0], color='dodgerblue', label='$E_g$'),
               Line2D([0],[0], color='orangered', label='$T_{2g}$'),
               #Patch(color='gray', alpha=0.3, label="$\Delta(\omega)$"),
               #Line2D([0],[0], color='k', marker='s', lw=0,  markersize=5, label="poles"),
               #Line2D([0],[0], color='k', label='$\Delta^{discr}_\eta(\omega)$')
               ]
ax.legend(handles=custom_leg1, loc="upper left") #bbox_to_anchor=(0.5, 0.6, 0.5, 0.5))
w_width = 21. #w_mesh[-1] - w_mesh[0]
w_center = -6 # (w_mesh[-1] + w_mesh[0])/2
fact = 1
if plot_horizontal:
    ax.set_xlim(w_center-fact*w_width/2, w_center+fact*w_width/2)
    #ax.set_xticks([-10, -5, 0, 5])
else:
    ax.set_ylim(w_center-fact*w_width/2, w_center+fact*w_width/2)
    #ax.set_yticks([-10, -5, 0, 5])
    #ax.get_xaxis().set_visible(False)
    #ax.spines['bottom'].set_visible(False)

#ax.vlines([w_mesh[0], w_mesh[-1]], *ax.get_ylim(), color='black', lw=0.5)
#ax.text(0.01, 0.92, "$N_{bins} = $"+f" {Nbins} eV", transform=ax.transAxes)
#ax.text(0.01, 0.80, "$\eta=\Delta_{bin}/2=$"+" {:.2f} eV".format(eta), transform=ax.transAxes)
#nbath_str = "$N_{bath}(E_g)=$"+"{}".format(Nbath_Eg)+"\n$N_{bath}(T_{2g})=$"+"{}".format(Nbath_T2g)
#ax.text(0.01, 0.55, nbath_str, transform=ax.transAxes)
#ax.set_title("Bath parametrization $N_{bins}=$"+f"{Nbins},   "+"$\\theta=$"+ f"{threshold}")

plt.tight_layout()
savename = "./defense_lin_range{}_{}_Nbins{}_thresholdV{}_Gauss{:.2f}_{}_mini.png".format(wmin, wmax, Nbins,threshold,eta, "horizontal" if plot_horizontal else "vertical")
plt.savefig(savename, dpi=300, bbox_inches='tight', transparent=True)
plt.show()
