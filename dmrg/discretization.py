import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from triqs.gf import *
from triqs.operators import *
from triqs_dft_tools.sumk_dft import *
from triqs.plot.mpl_interface import oplot

from h5 import *
import numpy as np


"""
discretization.py
-----------------
Utilities to discretize a continuous bath hybridization function V2_w
and to plot the continuous and discretized hybridization.

Notes:
- The function `discretisation` purposely places no discrete sites inside
  the energy window (0, gap) — i.e., it leaves the gap empty.
"""


def plot_disc(V2_w, V2_up, w_up, V2_dn, w_dn, w_bins, show=False):
    """Plot continuous hybridization and the resulting discretization.

    Args:
        V2_w: TRIQS Green's function-like object containing the continuous
            hybridization on a mesh. Expected to have spin components
            `"up_0"` and `"down_0"` with `.data` arrays indexed by mesh.
        V2_up (ndarray): Discretized |V|^2 values for the up spin (length M).
        w_up (ndarray): Discretized energies (centers) for up spin (length M).
        V2_dn (ndarray): Discretized |V|^2 values for the down spin (length M).
        w_dn (ndarray): Discretized energies (centers) for down spin (length M).
        w_bins (ndarray): Bin-edge energies used for discretization.
        show (bool): If True, call `plt.show()` after saving the figure.

    Returns:
        None. The figure is saved to `img/bath/discr_bath_N{N}.png` where N
        is `len(V2_up)`.

    Notes:
        - This function uses module-level globals `w_min_disc` and
          `w_max_disc` to set the x-limits of the plot.
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # continuous frequency mesh from the TRIQS object
    w = np.array([w for w in V2_w.mesh.values()])

    # draw the continuous hybridization (imaginary part already computed)
    ax.fill_between(w, V2_w["up_0"].data[:, 0, 0], color='dodgerblue', alpha=0.3, label="up")
    ax.fill_between(w, V2_w["down_0"].data[:, 0, 0], color='orangered', alpha=0.3, label="down")

    # draw discretized bath sites as bars
    ax.bar(w_up, V2_up, width=0.2, align="center", color='dodgerblue', label="discretize up")
    ax.bar(w_dn, V2_dn, width=0.2, align="center", color='orangered', label="discretize down")

    # mark the bin edges used for discretization
    ax.plot(w_bins, np.zeros(len(w_bins)), "k+", markersize=10, markeredgewidth=3)

    ax.set_xlim(w_min_disc - 1, w_max_disc + 1)
    ax.set_ylim(0,)
    ax.set_xlabel(r"$\omega \; [eV]$")
    ax.set_ylabel(r"$\left| V(\omega) \right|^2 \; [eV]$")

    ax.legend()

    fig.tight_layout()
    fig.savefig("img/bath/discr_bath_N%d.png" % len(V2_up), dpi=150)

    if show:
        plt.show()


def discretisation(V2_w, N_bins, w_min_disc, w_max_disc):
    """Discretize a continuous bath hybridization `V2_w` into a set of
    discrete bath sites, explicitly leaving the gap empty (no sites inside
    the interval (0, gap)).

    The discretization procedure:
    - Constructs `w_bins` by concatenating a set of bin edges from
      `w_min_disc` to 0 and from `gap` to `w_max_disc` so that the interval
      (0, gap) contains no bins and therefore no discrete sites.
    - For each bin [w_bins[i], w_bins[i+1]) it integrates the continuous
      |V(omega)|^2 over the bin to obtain the discrete weight `V2_*[i]` and
      computes the centroid `w_*[i]` as the first moment over the bin.

    Args:
        V2_w: TRIQS object containing the continuous hybridization on a mesh
            (spin components `"up_0"` and `"down_0"` expected).
        N_bins (int): Number of bin edges (the function produces N_bins-1
            discrete sites per spin). N_bins should be even so that
            half the bins are below zero and half above the `gap`.
        w_min_disc (float): Minimum energy for discretization (left-most).
        w_max_disc (float): Maximum energy for discretization (right-most).

    Returns:
        V2_up (ndarray): Discrete |V|^2 for up spin (length N_bins-1).
        w_up (ndarray): Energy centroids for up spin (length N_bins-1).
        V2_dn (ndarray): Discrete |V|^2 for down spin (length N_bins-1).
        w_dn (ndarray): Energy centroids for down spin (length N_bins-1).
        w_bins (ndarray): The bin-edge energies used for discretization
            (length N_bins).

    Notes:
        - The variable `gap` is taken from the module scope and determines
          the low edge of the upper set of bins (no bins placed in (0,gap)).
        - Integration is approximated by simple Riemann sums using the
          grid spacing `dw` from `V2_w.mesh`.
    """

    # create bin edges: half from w_min_disc..0, half from gap..w_max_disc
    if N_bins%2==0:
        w_bins = np.concatenate((np.linspace(w_min_disc, 0, int(N_bins / 2)+1),
                             np.linspace(gap, w_max_disc, int(N_bins / 2)+1)))
    else:
        w_bins = np.concatenate((np.linspace(w_min_disc, 0, int(N_bins / 2)+1),
                             np.linspace(gap, w_max_disc, int(N_bins / 2)+2)))
    # one more bath in conduction states if odd bc tail is bigger + need odd bath to have
    # even site for the impurity problem.

    # frequency mesh of continuous data and mesh spacing (assumes uniform)
    w = np.array([w for w in V2_w.mesh.values()])
    dw = w[1] - w[0]

    # arrays to hold discretized weights and centroids (one fewer than edges)
    V2_up = np.zeros(N_bins)
    w_up = np.zeros(N_bins)
    V2_dn = np.zeros(N_bins)
    w_dn = np.zeros(N_bins)

    # loop over bins and compute integrated weight and centroid for each spin
    i=0
    for j in range(N_bins+1):

        if j == int(N_bins/2): continue
        # find the nearest mesh indices for the left and right bin edges
        index_i = np.argmin(np.absolute(w - w_bins[j]))
        index_i1 = np.argmin(np.absolute(w - w_bins[j + 1]))

        # integrated |V|^2 over the bin (Riemann sum times dw)
        V2_up[i] = np.sum(V2_w["up_0"].data[index_i:index_i1]) * dw
        # centroid = first moment / integrated weight
        w_up[i] = (1.0 / V2_up[i]) * np.sum(w[index_i:index_i1] * V2_w["up_0"].data[index_i:index_i1].flatten()) * dw

        V2_dn[i] = np.sum(V2_w["down_0"].data[index_i:index_i1]) * dw
        w_dn[i] = (1.0 / V2_dn[i]) * np.sum(w[index_i:index_i1] * V2_w["down_0"].data[index_i:index_i1].flatten()) * dw

        i+= 1

    # save discretized bath to file: columns are V2_up, w_up, V2_dn, w_dn
    np.savetxt("bath_data/disc_bath_N%d.dat" % (N_bins),
               np.column_stack((V2_up, w_up, V2_dn, w_dn)),
               header="|V2_up|^2  w_up  |V2_dn|^2  w_dn", fmt="%.8e")

    return V2_up, w_up, V2_dn, w_dn, w_bins
        
##################################################
###############    Parameters    #################
##################################################

N_bain_tab = np.arange(4, 30)
gap = 1.75
w_min_disc = -3.

##################################################
##################    main    ####################
##################################################

with HDFArchive("MaxEnt_data/maxent_U4.00_beta100.00_Sigma_w_ISC.h5", "r") as A :
    Sigma_w = A["Sigma_w"]
    Sigma_w = Sigma_w

with HDFArchive("MaxEnt_data/maxent_U4.00_beta100.00_G_w.h5", "r") as A :
    G_w = A["G_w"]

# continuous hybridization
V2_w = (1/np.pi*(inverse(G_w) + Sigma_w)).imag

# symmetrization over spin
V2_w["up_0"] << (V2_w["up_0"]+V2_w["down_0"])/2
V2_w["down_0"] << V2_w["up_0"]

for i, N_bain in enumerate(N_bain_tab):

    if N_bain%2==0:
        w_max_disc = gap + abs(w_min_disc)
    else:
        w_max_disc = gap + abs(w_min_disc) * (1+1/(int(N_bain/2)))

    V2_up, w_up, V2_dn, w_dn, w_bins = discretisation(V2_w, N_bain, w_min_disc, w_max_disc)
    plot_disc(V2_w, V2_up, w_up, V2_dn, w_dn, w_bins, show=False)