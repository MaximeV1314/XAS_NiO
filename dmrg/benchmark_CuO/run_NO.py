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
# N_bain_tab = [24]

epsilon = 1e-7
tridiag_method = "LA"
truncate = False

init_state_method = "up"
# folder_dm = "dmrg_data/lc_updown/"
folder_dm_up = "dmrg_data/up/"
folder_dm_down = "dmrg_data/down/"

#######################################################
##################      Main       ####################
#######################################################


for i, N_bain in enumerate(N_bain_tab):

    bain = np.loadtxt("bath_data/disc_bath_N%d.dat"%N_bain).T
    V_l = np.sqrt(bain[0])
    e_l = bain[1]

    dm_up = np.loadtxt(folder_dm_up + "N%d/density_matrix_N%d.dat"%(N_bain, N_bain), dtype = np.complex64)
    dm_dn = np.loadtxt(folder_dm_down + "N%d/density_matrix_N%d.dat"%(N_bain, N_bain), dtype = np.complex64)
    dm = 1/2*dm_up + 1/2*dm_dn

    lat_AIM = ptn.mp.lat.u1u1.genFermiHubbard(N_bain+1) # N_bain + imp

    L = lat_AIM.size()
    if L%2==0: imp_index = int(L/2)-1
    else: imp_index = int(L/2)
    init_state = init_state_naive(lat_AIM, e_l, imp_index, init_state_method, occ_deriv = 1, mag = 0)

    hamiltonian_NO(lat_AIM, U, mu, ed, V_l, e_l, dm, imp_index, epsilon=epsilon, tridiag_method=tridiag_method, truncate=truncate)
    # dmrg_run(init_state, lat_AIM, "dmrg_data/%s/N%d_NO/"%(init_state_method, N_bain))
