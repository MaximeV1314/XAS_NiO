import pyten as ptn
import numpy as np
from main import *

#######################################################
####################   FONCTIONS   ####################
#######################################################


#######################################################
####################   PARAMETERS  ####################
#######################################################

U = 4.
mu = 1.166059
ed = -0.042731
N_bain_tab = np.arange(4, 30)
N_bain_tab = [11, 13, 16, 26]
init_state_method = "random"
# folder_dm = "dmrg_data/lc_updown/"
folder_dm = "dmrg_data/sebastian/"

#######################################################
##################      Main       ####################
#######################################################


for i, N_bain in enumerate(N_bain_tab):

    bain = np.loadtxt("bath_data/disc_bath_N%d.dat"%N_bain).T
    V_l = np.sqrt(bain[0])
    e_l = bain[1]

    dm = np.loadtxt(folder_dm + "N%d/density_matrix_N%d.dat"%(N_bain, N_bain), dtype = np.complex64)

    lat_AIM = ptn.mp.lat.u1u1.genFermiHubbard(N_bain+1) # N_bain + imp

    L = lat_AIM.size()
    if L%2==0: imp_index = int(L/2)-1
    else: imp_index = int(L/2)
    init_state = init_state_naive(lat_AIM, e_l, imp_index, init_state_method, occ_deriv = 1, mag = 0)

    hamiltonian_NO(lat_AIM, U, mu, ed, V_l, e_l, dm, imp_index)
    # dmrg_run(init_state, lat_AIM, "dmrg_data/%s/N%d_NO/"%(init_state_method, N_bain))
