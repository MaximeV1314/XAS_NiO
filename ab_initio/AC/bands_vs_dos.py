import numpy as np
import matplotlib.pyplot as plt

from triqs.gf import *
from triqs.gf.tools import *
from triqs.operators import *
from triqs_dft_tools.sumk_dft import *
from triqs_dft_tools.sumk_dft_tools import *


def k_path_extraction(kgrid):
    """
    Extrait les points k de kgrid appartenant au chemin kpath donné ci-dessous.
    
    Paramètres:
    kgrid : points k Monkhorst Pack donnés par QE nscf [array Nk x 3]

    Retourne:
    kpath_indices : array d'indices correspondant aux points k du chemin [array 1D]
    kpath_points  : array correspondant aux points k du chemin [array 1D x 3]
    """

    tol = 1e-8

    # -------- k-path --------
    kpath = [
        ("W", [-0.250, 0.500, 0.250], "L", [0.000, 0.500, 0.000]),
        ("L", [0.000, 0.500, 0.000], "G", [0.000, 0.000, 0.000]),
        ("G", [0.000, 0.000, 0.000], "X", [0.000, 0.500, 0.500]),
        ("X", [0.000, 0.500, 0.500], "W", [-0.250, 0.500, 0.250]),
        ("W", [-0.250, 0.500, 0.250], "K", [-0.375, 0.375, 0.000]),
    ]


    # ---------- utils périodiques [0,1] ----------

    def wrap01(k):
        return np.mod(k, 1.0)


    def periodic_diff01(k1, k2):
        """différence minimale périodique sur [0,1]"""
        d = k1 - k2
        return d - np.round(d)


    def find_k_on_segment_periodic(k1, k2, kgrid):

        k1 = wrap01(np.array(k1))
        k2 = wrap01(np.array(k2))

        direction = periodic_diff01(k2, k1)

        selected = []

        for ik, k in enumerate(kgrid):

            k = wrap01(k)

            diff = periodic_diff01(k, k1)

            if np.linalg.norm(direction) < tol:
                continue

            t = np.dot(diff, direction) / np.dot(direction, direction)

            if -tol <= t <= 1+tol:

                k_on_line = wrap01(k1 + t * direction)

                if np.linalg.norm(periodic_diff01(k, k_on_line)) < tol:
                    selected.append((t, ik, k))

        selected.sort(key=lambda x: x[0])

        return selected


    # -------- construction du k-path --------

    kpath_points = []
    kpath_indices = []
    labels = []

    for start_label, k1, end_label, k2 in kpath:

        seg = find_k_on_segment_periodic(k1, k2, kgrid)

        if len(kpath_points) != 0:
            seg = seg[1:]

        for t, ik, k in seg:
            kpath_points.append(k)
            kpath_indices.append(ik)

        labels.append(start_label)

    labels.append(kpath[-1][2])

    kpath_points = np.array(kpath_points)
    kpath_indices = np.array(kpath_indices)

    print("Indices du k-path:")
    print(kpath_indices)

    print("\nPoints du k-path:")
    print(kpath_points)

    return kpath_indices, kpath_points


def data_loader(fname):
    
    data = np.loadtxt(fname)
    energy = data[:, 0]
    pdos = data[:, 1]  # ldos col, total contribution for a given orbital
    ppdos = data[:, 2:]

    return energy, pdos, ppdos

#-----------------------------------------------------------------
#                       DFT Data loading
#-----------------------------------------------------------------

data_dos = np.loadtxt("../../DFT/pdos/NiO.pdos_tot").T
data_bands = np.loadtxt("../../DFT/bands.out.gnu").T.reshape(2, 16, 181) # 62    184

energies, pdos3s_Ni, ppdos3s_Ni =  data_loader("../../DFT/pdos/NiO.pdos_atm#1(Ni)_wfc#1(s)")
energies, pdos3p_Ni, ppdos3p_Ni =  data_loader("../../DFT/pdos/NiO.pdos_atm#1(Ni)_wfc#2(p)")
energies, pdos3d_Ni, ppdos3d_Ni =  data_loader("../../DFT/pdos/NiO.pdos_atm#1(Ni)_wfc#3(d)")
energies, pdos4s_Ni, ppdos4s_Ni =  data_loader("../../DFT/pdos/NiO.pdos_atm#1(Ni)_wfc#4(s)")

energies, pdos2s_O, ppdos2s_O =  data_loader("../../DFT/pdos/NiO.pdos_atm#2(O)_wfc#1(s)")
energies, pdos2p_O, ppdos2p_O =  data_loader("../../DFT/pdos/NiO.pdos_atm#2(O)_wfc#2(p)")

Efermi = 14.0223

Emax = 5
Emin = -10
dos_max = 10

#-----------------------------------------------------------------
#                       DMFT data loading
#-----------------------------------------------------------------

DMFT_folder = "../../DMFT/DMFT_NiO_CTSEG/"

with HDFArchive("MaxEnt_data/maxent_U7.00_beta40.00_Sigma_w_ISC.h5", "r") as A :
    Sigma_w = A["Sigma_w"]

with HDFArchive("MaxEnt_data/maxent_U7.00_beta40.00_G_w.h5", "r") as A :
    G_w = A["G_w"]

with HDFArchive(DMFT_folder+"DMFT_data/results/DMFT_U7.00_beta40.00.h5", "r") as A :
    it = len(A) - 2
    mu    = A["iteration_%d"%it]["mu"]
    dc_energ = A["iteration_%d"%it]["dc_energ"]
    dc_imp   = A["iteration_%d"%it]["dc_imp"]

SK = SumkDFTTools(hdf_file = DMFT_folder + 'w90_to_triqs.h5', mesh = Sigma_w.mesh, beta = 40., use_dft_blocks=True)

SK.set_Sigma([Sigma_w])
SK.set_mu(mu)
SK.set_dc(dc_imp,dc_energ)

#########   DMFT DOS    ########
dmft_dos, dmft_pdos, dmft_ppdos = SK.density_of_states(mu=mu, proj_type="wann", save_to_file=False)    # dos, pdos

#########   DMFT bands    ########

omega = np.array([w  for w in Sigma_w.mesh.values()])
idx_min = np.argmin(np.abs(omega-Emin))
idx_max = np.argmin(np.abs(omega-Emax))

with HDFArchive(DMFT_folder+'w90_to_triqs.h5', "r") as A :
    kpts = A["dft_input"]["kpts"]
kpath_indices, kpath_points = k_path_extraction(kpts)
k_path_len = len(kpath_indices)
omega = np.array([w  for w in Sigma_w.mesh.values()])

A_k = np.zeros((k_path_len, len(omega)))
for i, ik in enumerate(kpath_indices):
    G_ki = SK.lattice_gf(ik=ik)
    for j in range(8):
        A_k[i] += G_ki["up"][j,j].data.imag

A_k *= -2/np.pi

"""
# interpolation of A_k
N_interpo = 100 # linear interpolation between 2 k_point
A_k_interpo = np.zeros((N_interpo*k_path_len, len(omega)))
for i in range(k_path_len-1):
    for j in range(N_interpo):
        A_k_interpo[i*N_interpo+j] = (1-j/N_interpo) * A_k[i] + j/N_interpo * A_k[i+1]

A_k = A_k_interpo
"""

#-----------------------------------------------------------------
#                           Plot
#-----------------------------------------------------------------

fig, ax = plt.subplots(1,2, figsize = (14, 8), gridspec_kw={"width_ratios": [1, 2], "wspace":0.05})

#----------------------   First figure   ------------------------#

##########   DFT DOS   ############

mask = data_dos[0] - Efermi < 0
#ax[0].plot(data_dos[1], data_dos[0] - Efermi, "-", color = "black", linewidth = 3, label = "DOS")
#ax[0].fill_betweenx(data_dos[0][mask] - Efermi, data_dos[1][mask], 0, color = "black", alpha = 0.15)
ax[0].fill_betweenx(data_dos[0] - Efermi, data_dos[1], 0, color = "black", alpha = 0.15, label = "DFT")

# PDOS
#ax[0].plot(pdos3d_Ni, energies - Efermi, color = "red", label = "Ni 3d")
#ax[0].plot(2*ppdos3d_Ni[:,0], energies - Efermi, color = "darkred", label = r"Ni $e_g$")
#ax[0].plot(3*ppdos3d_Ni[:,1], energies - Efermi, color = "orange", label = r"Ni $t_{2g}$")

#ax[0].plot(pdos2p_O, energies - Efermi, color = "blue", label = "O  2p")

##########   DMFT DOS   ############

ax[0].plot(dmft_dos["up"]+dmft_dos["down"], omega, "-k", lw = 3, label = "DMFT")

ax[0].plot(2*(dmft_ppdos[0]["up"][:,0,0]+dmft_ppdos[0]["down"][:,0,0]), omega, "-", color = 'dodgerblue', lw=3, label = r"$e_g$")
ax[0].plot(3*(dmft_ppdos[0]["up"][:,1,1]+dmft_ppdos[0]["down"][:,1,1]), omega, "-", color = 'orangered', lw=3, label = r"$t_{2g}$")
ax[0].plot(dmft_dos["up"]+dmft_dos["down"] - 3*(dmft_ppdos[0]["up"][:,1,1]+dmft_ppdos[0]["down"][:,1,1])\
                -2*(dmft_ppdos[0]["up"][:,0,0]+dmft_ppdos[0]["down"][:,0,0]), omega, \
                    "-", color = 'green', lw=3, label = r"O 2p")

############ Plot Parameters ###########

ax[0].plot([35, 0], [0, 0], ":", color = "black")

ax[0].set_xlim(dos_max, 0)
ax[0].set_ylim(Emin, Emax)
ax[0].set_xticks([])
ax[0].set_xlabel("DOS", fontsize = 24)
ax[0].set_ylabel("Energy [eV]", fontsize = 24)

ax[0].tick_params(axis='both', labelsize=20)
ax[0].legend(fontsize = 20)

#---------------------   Second figure  -------------------------#

sym_kpoints = [0.0, 0.7071, 1.5731, 2.5731, 3.0731, 3.4267]
sym_kpoints_name = ["W", "L", "$\Gamma$", "X", "W", "K"]

####### DMFT bands #######
im = ax[1].imshow(A_k[:, idx_min:idx_max].T, aspect='auto', origin='lower', norm = "log", interpolation="none", vmin = 0.1, vmax=10,\
             extent=[sym_kpoints[0], sym_kpoints[-2], Emin, Emax])
fig.colorbar(im, label = r"$A(k,\omega)$")

####### DFT bands #######
for i in range(len(data_bands[0, :])):
    ax[1].plot(data_bands[0, i], data_bands[1, i] - Efermi, "-", color = "white")
ax[1].set_ylim(Emin, Emax)
ax[1].set_xlim(data_bands[0,0,0], data_bands[0,0,-1])

ax[1].plot([0, sym_kpoints[-1]], [0, 0], ":", color = "white")                                             # fermi
for i in range(len(sym_kpoints)):
    ax[1].plot([sym_kpoints[i], sym_kpoints[i]], [Emin, Emax], color = "black", linewidth = 0.4)     # lignes verticales

####### plot parameters #######
ax[1].set_xticks(sym_kpoints)               
ax[1].set_xticklabels(sym_kpoints_name, fontsize = 24) 
ax[1].set_yticks([])   

####### save and plot ########
fig.savefig("dos_vs_bands_in.pdf", bbox_inches='tight', transparent=True)
plt.show()