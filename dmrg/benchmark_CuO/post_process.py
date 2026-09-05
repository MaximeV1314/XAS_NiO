import pyten as ptn
import numpy as np
import matplotlib.pyplot as plt
import os

#######################################################
####################   FONCTIONS   ####################
#######################################################

def density_matrix_plot(density_matrix, imp_index):

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    im = ax.imshow(density_matrix.real, extent=[0, L, L, 0], vmin=-1, vmax=1, cmap="bwr")
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

    ax.set_title("bond dim = %d"%(bond_dim))    # S^2_imp = %.2f

    plt.tight_layout()
    plt.savefig(savedir + "density_matrix_N%d.png"%(N_bain))
    plt.close()


def occ_mag_plot(occ, mag):

    fig, ax = plt.subplots(1, 2, figsize = (12, 5))

    ax[0].plot(occ, "o")
    ax[0].set_title("Tot. occ = %.2f"%np.sum(occ))
    ax[0].set_xlabel("sites [%d = imp]"%(int(N_bain/2)) )
    ax[0].set_ylabel(r"$\leftangle n \rightangle$")
    ax[0].set_ylim(-0.1, 2.1)

    ax[1].plot(mag, "o")
    ax[1].set_title("Tot. mag = %.2f"%np.sum(mag))
    ax[1].set_xlabel("sites [%d = imp]"%(int(N_bain/2)) )
    ax[1].set_ylabel(r"$\leftangle s_z \rightangle$")
    ax[1].set_ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig(savedir + "occ_mag_N%d.png"%(N_bain))
    plt.close()

#######################################################
####################   PARAMETERS  ####################
#######################################################

N_bain_tab = np.arange(4, 30)
# N_bain_tab = [10]
init_state_method = "random"    # up, down, singlet, triplet, 
hamitlonian_type  = "_NO"    # empty = star rep, _ch = chaine rep, _ch_v2 = chaine rep with numpy (faster), _NO = natural orbitals

#######################################################
##################      Main       ####################
#######################################################

E0_tab  = np.zeros(len(N_bain_tab))     # GS per spin

for i, N_bain in enumerate(N_bain_tab):
    print(i/len(N_bain_tab))

    savedir = "img/GS/%s%s/"%(init_state_method, hamitlonian_type)
    os.makedirs(savedir, exist_ok=True)

    calcdir = "dmrg_data/%s/N%d%s/"%(init_state_method, N_bain, hamitlonian_type)
    lat_AIM = ptn.mp.Lattice(calcdir + "lattice.lat")
    L = lat_AIM.size()
    if hamitlonian_type == "_ch" or hamitlonian_type == "_ch_v2":
        imp_index=0
    else:
        if L%2==0: imp_index = int(L/2)-1
        else: imp_index = int(L/2)

    state = ptn.mp.MPS(calcdir+"final_state.mps")
    bond_dim = max(len(sublist) for sublist in ptn.mp.schmidt_values(state))
    density_matrix = np.zeros((L, 2, L, 2), dtype = np.complex64)

    #S2_imp = np.array(ptn.mp.local_expectation(lat_AIM, state, "s")[0])[imp_index].real
    #S2_imp = ptn.mp.expectation(state, lat_AIM.get("s", imp_index) * lat_AIM.get("s", imp_index)).real

    for j in range(L):
        for k in range(L):

            density_matrix[j, 0, k, 0] = ptn.mp.expectation(state, lat_AIM.get("cu", j) * lat_AIM.get("chu", k))
            density_matrix[j, 0, k, 1] = ptn.mp.expectation(state, lat_AIM.get("cu", j) * lat_AIM.get("chd", k))
            density_matrix[j, 1, k, 0] = ptn.mp.expectation(state, lat_AIM.get("cd", j) * lat_AIM.get("chu", k))
            density_matrix[j, 1, k, 1] = ptn.mp.expectation(state, lat_AIM.get("cd", j) * lat_AIM.get("chd", k))
    
    np.savetxt("dmrg_data/%s/N%d%s/density_matrix_N%d.dat"%(init_state_method, N_bain, hamitlonian_type, N_bain), \
                density_matrix.reshape(2*L, 2*L))
    density_matrix_plot(density_matrix.reshape(2*L, 2*L), imp_index)
    
    if L%2==0: n_val = int(L/2)
    else: n_val = int(L/2)+1
    # E0_tab[i] = ptn.mp.expectation(state, lat_AIM.get("H")).real/n_val
    E0_tab[i] = ptn.mp.expectation(state, lat_AIM.get("H")).real
    print("N_bath = %d, E0 = [eV] "%N_bain, E0_tab[i])

    occ = np.array(ptn.mp.local_expectation(lat_AIM, state, "n")[0]).real
    mag = np.array(ptn.mp.local_expectation(lat_AIM, state, "sz")[0]).real
    occ_mag_plot(occ, mag)

fig, ax = plt.subplots(1, 1, figsize = (6, 6))

ax.plot(N_bain_tab, E0_tab, "o")
ax.set_xlabel("Number of bath")
ax.set_ylabel(r"$\leftangle \hat{H} \rightangle/L$")

plt.tight_layout()
plt.savefig(savedir + "GS_vs_bath.png")
