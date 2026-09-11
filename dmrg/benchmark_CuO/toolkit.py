import pyten as ptn
import numpy as np
import matplotlib.pyplot as plt
import glob, os, shutil
import warnings
from matplotlib.colors import LogNorm

####################################################################
####################################################################
####################                         #######################
####################        Plot kit         #######################
####################                         #######################
####################################################################
####################################################################

def density_matrix_plot(density_matrix, imp_index, spin=True):
    """
    Plot the density matrix with imshow in regular scale (first plot) and log scale (second plot).

    density_matrix : density matrix (real matrix)
    imp_index : impurity index (integer)
    spin : if there is spin or not. If so, the density matrix is a matrix of 2x2 blocks.
    """
        
    L = len(density_matrix)

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    im = ax.imshow(density_matrix.real, extent=[0, L, L, 0], vmin=-1, vmax=1, cmap="bwr")
    
    if (type(imp_index) == type(0) or type(imp_index) == type(1.1)) and spin==False :
        ax.hlines([imp_index, imp_index+1], xmin = 0, xmax = L, color = "red")
        ax.vlines([imp_index, imp_index+1], ymin = 0, ymax = L, color = "red")
    elif (type(imp_index) == type(0) or type(imp_index) == type(1.1)) and spin==True :
        ax.hlines([2*imp_index, 2*imp_index+2], xmin = 0, xmax = L, color = "red")
        ax.vlines([2*imp_index, 2*imp_index+2], ymin = 0, ymax = L, color = "red")

    cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
    cbar.set_label(r"Occupation $c_i^{\dagger}c_j$", rotation=270, fontsize = 22, labelpad=30)
    cbar.ax.tick_params(labelsize=18) 

    ax.set_xlabel("$i$ site", fontsize = 22)
    ax.set_ylabel("$j$ site", fontsize = 22)

    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if spin == False:
        ax.set_xticks(np.arange(0, L))
        ax.set_yticks(np.arange(0, L))
    elif spin == True :
        ticks = np.arange(0, L+1, 2)
        labels = np.arange(0, int(L/2)+1)

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)

    plt.tight_layout()
    # plt.savefig(savedir + "density_matrix_N%d.png"%(N_bain))
    plt.show()

    plt.close()


    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    im = ax.imshow(np.abs(density_matrix.real), norm=LogNorm(vmin=0.00001, vmax=1.), extent=[0, L, L, 0], cmap="viridis")
    
    if (type(imp_index) == type(0) or type(imp_index) == type(1.1)) and spin==False :
        ax.hlines([imp_index, imp_index+1], xmin = 0, xmax = L, color = "red")
        ax.vlines([imp_index, imp_index+1], ymin = 0, ymax = L, color = "red")
    elif (type(imp_index) == type(0) or type(imp_index) == type(1.1)) and spin==True :
        ax.hlines([2*imp_index, 2*imp_index+2], xmin = 0, xmax = L, color = "red")
        ax.vlines([2*imp_index, 2*imp_index+2], ymin = 0, ymax = L, color = "red")

    cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
    cbar.set_label(r"Occupation $c_i^{\dagger}c_j$", rotation=270, fontsize = 22, labelpad=30)
    cbar.ax.tick_params(labelsize=18) 

    ax.set_xlabel("$i$ site", fontsize = 22)
    ax.set_ylabel("$j$ site", fontsize = 22)

    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if spin == False:
        ax.set_xticks(np.arange(0, L))
        ax.set_yticks(np.arange(0, L))
    elif spin == True :
        ticks = np.arange(0, L+1, 2)
        labels = np.arange(0, int(L/2)+1)

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)

    plt.tight_layout()
    # plt.savefig(savedir + "density_matrix_N%d.png"%(N_bain))
    plt.show()

def hamiltonian_plot(H, site_dict, title="", file=""):
    """
    Plot the Hamiltonian with imshow in regular scale (first plot) and log scale (second plot).

    H : hamiltonian (real matrix)
    site_dict : dictionnary of type {int1:str1, int2:str2, ...} with the integer indicating the # site
                and the string indicating the name of the site.
    title : title of the plot.
    file  : name of the save images. If the string is empty, it does't save the pictures.
    """

    L = len(H)

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    im = ax.imshow(H, extent=[0, L, L, 0], vmin=-2, vmax=2, cmap="PiYG")
    #if type(imp_index) == type(0) or type(imp_index) == type(1.1):
     #   print("prout")
    
    color = ["red", "green", "blue", "brown"]

    for k, (site, name) in enumerate(site_dict.items()):
        if k==0 : zorder = 100
        else : zorder = 99
        ax.hlines([site, site+1], xmin = 0, xmax = L, color = color[k], label = name, zorder = zorder)
        ax.vlines([site, site+1], ymin = 0, ymax = L, color = color[k], zorder = zorder)
    
    cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
    cbar.set_label(r"Energy [eV]", rotation=270, fontsize = 22, labelpad=30)
    cbar.ax.tick_params(labelsize=18) 

    ax.set_xlabel("$i$ site", fontsize = 22)
    ax.set_ylabel("$j$ site", fontsize = 22)

    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xticks(np.arange(0, L))
    ax.set_yticks(np.arange(0, L))

    ax.set_title(title, fontsize=22)
    ax.legend()

    plt.tight_layout()
    if len(file) > 0:
        plt.savefig("%s.png"%(file))
    else:
        plt.show()

    plt.close()

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    im = ax.imshow(np.abs(H), extent=[0, L, L, 0], norm=LogNorm(vmin=0.0001, vmax=1., clip=True), cmap="viridis")

    for k, (site, name) in enumerate(site_dict.items()):
        if k==0 : zorder = 100
        else : zorder = 99
        ax.hlines([site, site+1], xmin = 0, xmax = L, color = color[k], label = name, zorder = zorder)
        ax.vlines([site, site+1], ymin = 0, ymax = L, color = color[k], zorder = zorder)
    
    cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
    cbar.set_label(r"Energy [eV]", rotation=270, fontsize = 22, labelpad=30)
    cbar.ax.tick_params(labelsize=18) 

    ax.set_xlabel("$i$ site", fontsize = 22)
    ax.set_ylabel("$j$ site", fontsize = 22)

    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xticks(np.arange(0, L))
    ax.set_yticks(np.arange(0, L))

    ax.set_title(title, fontsize=22)
    ax.legend()

    plt.tight_layout()
    if len(file) > 0:
        plt.savefig("%s_log.png"%(file))
    else:
        plt.show()
        
####################################################################
####################################################################
####################                         #######################
####################       Lanczos kit       #######################
####################                         #######################
####################################################################
####################################################################

def lanczos_full(v, A):

    """
    Tridiagonalisation of a matrix 'A' giving a initial vector 'v' using the Lanczos algorithm.
    This produces an orthonormal basis 'basis' that tridiagonalizes 'A' with matrix 'T'.

    Parameters:
        v: initial state.
        A: matrix to tridiagonalize

    Returns:
        T : the tridiagonal matrix
        basis : the basis
    """

    N = len(v)
    vl = v.copy()   # energy vector
    vl /= np.linalg.norm(vl)
    vl_previous = np.zeros(N)

    alpha = np.zeros(N)
    beta  = np.zeros(N)

    basis = [vl]

    for l in range(N-1):
        wl = A @ vl - beta[l] * vl_previous
        alpha[l] = vl.T @ wl

        wl -= alpha[l] * vl
        beta[l+1] = np.linalg.norm(wl)

        vl_previous = vl.copy()
        vl = wl / np.linalg.norm(wl)
        basis.append(vl)

        # Check orthogonality: verify that new state is orthogonal to all previous states
        for n in range(len(basis)-1):
            overlap_basis = np.abs(basis[n].T @ vl)
            if overlap_basis > 1e-8:
                warnings.warn("<b%d|b%d> = %.2E > 1e-8"%(n+1, l+1, overlap_basis))
        
        # Check normalization: verify that new basis state is normalized
        if abs(np.abs(np.linalg.norm(vl)) - 1) > 1e-6:
            warnings.warn("| <b%d|b%d> - 1 | = %.2E > 1e-6"%(l+1, l+1, overlap_basis))

    # Compute the final diagonal energy element
    wl = A @ vl - beta[-1] * vl_previous
    alpha[-1] = vl.T @ wl

    T = np.diag(alpha) + np.diag(beta[1:], k=1) + + np.diag(beta[1:], k=-1)
    basis = np.array(basis).T
    return T, basis
