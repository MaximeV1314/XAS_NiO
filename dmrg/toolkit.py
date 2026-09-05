import pyten as ptn
import numpy as np
import matplotlib.pyplot as plt
import glob, os, shutil
import warnings

####################################################################
####################################################################
####################                         #######################
####################        Plot kit         #######################
####################                         #######################
####################################################################
####################################################################

def density_matrix_plot(density_matrix, imp_index):

    L = len(density_matrix)

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    im = ax.imshow(density_matrix.real, extent=[0, L, L, 0], vmin=-1, vmax=1, cmap="bwr")
    if type(imp_index) == type(0) or type(imp_index) == type(1.1):
        ax.hlines([imp_index, imp_index+1], xmin = 0, xmax = L, color = "red")
        ax.vlines([imp_index, imp_index+1], ymin = 0, ymax = L, color = "red")
    
    cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
    cbar.set_label(r"Occupation $c_i^{\dagger}c_j$", rotation=270, fontsize = 22, labelpad=30)
    cbar.ax.tick_params(labelsize=18) 

    ax.set_xlabel("$i$ site", fontsize = 22)
    ax.set_ylabel("$j$ site", fontsize = 22)

    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xticks(np.arange(0, L))
    ax.set_yticks(np.arange(0, L))

    plt.tight_layout()
    # plt.savefig(savedir + "density_matrix_N%d.png"%(N_bain))
    plt.show()

def hamiltonian_plot(H, site_dict, title="", file=""):

    L = len(H)

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    im = ax.imshow(H, extent=[0, L, L, 0], vmin=-2, vmax=2, cmap="PiYG")
    #if type(imp_index) == type(0) or type(imp_index) == type(1.1):
     #   print("prout")
    
    color = ["red", "green", "blue", "brown"]

    for k, (site, name) in enumerate(site_dict.items()):
        ax.hlines([site, site+1], xmin = 0, xmax = L, color = color[k], label = name, zorder = 1000)
        ax.vlines([site, site+1], ymin = 0, ymax = L, color = color[k], zorder = 1000)
    
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
        plt.savefig("%s"%(file))
    else:
        plt.show()

####################################################################
####################################################################
####################                         #######################
####################       Lanczos kit       #######################
####################                         #######################
####################################################################
####################################################################