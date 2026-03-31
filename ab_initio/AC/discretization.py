import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from triqs.gf import *
from triqs.operators import *
from triqs_dft_tools.sumk_dft import *
from triqs.plot.mpl_interface import oplot

from h5 import *
import numpy as np

def plot_disc(V2_w, V2_eg, w_eg, V2_t2g, w_t2g, w_bins):

    fig, ax = plt.subplots(1, 1, figsize = (10, 6))

    w = np.array([w  for w in V2_w.mesh.values()])

    ax.fill_between(w, V2_w["up_0"].data[:,0,0], color = 'dodgerblue', alpha = 0.3, label = "eg")
    ax.fill_between(w, V2_w["up_1"].data[:,0,0], color = 'orangered', alpha = 0.3, label = "t2g")

    ax.bar(w_eg, V2_eg, width=0.2, align="center", color = 'dodgerblue', label = "discretize eg")
    ax.bar(w_t2g, V2_t2g, width=0.2, align="center", color = 'orangered', label = "discretize t2g")

    ax.plot(w_bins, np.zeros(len(w_bins)), "k+", markersize = 10, markeredgewidth=3)

    ax.set_xlim(w_min_disc-1, w_max_disc+1)
    ax.set_ylim(0,)
    ax.set_xlabel(r"$\omega \; [eV]$")
    ax.set_ylabel(r"$\left| V(\omega) \right|^2 \; [eV]$")

    ax.legend()

    fig.tight_layout()
    fig.savefig("discr_bath_U7.00_beta40.00.png", dpi = 150)

    plt.show()

def discretisation(V2_w, N_bins, w_min_disc, w_max_disc):

    w_bins = np.linspace(w_min_disc, w_max_disc, N_bins)
    w  = np.array([w  for w in V2_w.mesh.values()])
    dw = w[1] - w[0]

    V2_eg = np.zeros(N_bins-1)
    w_eg = np.zeros(N_bins-1)
    V2_t2g = np.zeros(N_bins-1)
    w_t2g = np.zeros(N_bins-1)

    for i in range(N_bins-1):
        index_i, index_i1 = np.argmin(np.absolute(w - w_bins[i])), np.argmin(np.absolute(w - w_bins[i+1]))

        V2_eg[i]  = np.sum(V2_w["up_0"].data[index_i:index_i1])*dw
        w_eg[i]  = 1/V2_eg[i] * np.sum(w[index_i:index_i1] * V2_w["up_0"].data[index_i:index_i1].flatten())*dw

        V2_t2g[i] = np.sum(V2_w["up_1"].data[index_i:index_i1])*dw
        w_t2g[i] = 1/V2_t2g[i] * np.sum(w[index_i:index_i1] * V2_w["up_1"].data[index_i:index_i1].flatten())*dw

    np.savetxt("disc_bath.dat", np.column_stack((V2_eg, w_eg, V2_t2g, w_t2g)), header="|V2_eg|^2  w_eg  |V2_t2g|^2  w_t2g", fmt="%.8e")
    return V2_eg, w_eg, V2_t2g, w_t2g, w_bins
        
##################################################
###############    Parameters    #################
##################################################

N_bins = 10
w_min_disc = -10.
w_max_disc = 5.

##################################################
##################    main    ####################
##################################################

with HDFArchive("MaxEnt_data/maxent_U7.00_beta40.00_Sigma_w_ISC.h5", "r") as A :
    Sigma_w = A["Sigma_w"]
    Sigma_w = Sigma_w

with HDFArchive("MaxEnt_data/maxent_U7.00_beta40.00_G_w.h5", "r") as A :
    G_w = A["G_w"]

# continuous hybridization
V2_w = (1/np.pi*(inverse(G_w) + Sigma_w)).imag

# symmetrisation over t2g and eg
for spin in ["up", "down"] :
    V2_w["%s_0"%spin] << (V2_w["%s_0"%spin] + V2_w["%s_3"%spin])/2
    V2_w["%s_3"%spin] << V2_w["%s_0"%spin]
    V2_w["%s_1"%spin] << (V2_w["%s_1"%spin] + V2_w["%s_2"%spin] + V2_w["%s_4"%spin])/3
    V2_w["%s_2"%spin] << V2_w["%s_1"%spin]
    V2_w["%s_4"%spin] << V2_w["%s_1"%spin]

V2_eg, w_eg, V2_t2g, w_t2g, w_bins = discretisation(V2_w, N_bins, w_min_disc, w_max_disc)
plot_disc(V2_w, V2_eg, w_eg, V2_t2g, w_t2g, w_bins)