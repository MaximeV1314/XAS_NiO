from triqs_dft_tools.sumk_dft import *
from triqs.gf import *
from triqs.operators import *
from triqs.operators.util import *
from triqs.atom_diag import *
from h5 import *

from triqs_cthyb import Solver
from triqs_cthyb.tail_fit import tail_fit
import sys

import time
from datetime import datetime
import numpy as np
import os
import pandas as pa

pa.set_option("display.max_columns", None)
pa.set_option("display.width", None)
pa.set_option("display.max_colwidth", None)

if not mpi.is_master_node():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
 	
###########################################################################################
###########################             Functions           ###############################
###########################################################################################

def DMFT(beta : float, U : float, J : float, mix : float, mix_vary:bool, n_l : int, n_iw : int, n_tau : int, \
         n_cycles : int, n_warmup : int, length_cycles : int, n_loops : int, threshold : float,\
         fit : bool, fit_min_n : float, fit_max_n : float, \
         file_name : str, folder : str = "", measure_density : bool = False, \
         file_app = False, mu_init = 0, mu_dc = 0):

    """
    DMFT loop
    Parameters
    ----------
    beta         : temperature           [float]
    U            : interaction term      [float]
    J            : hund term             [float]
    mix          : mix of self and G     [float 0-1]
    mix_vary     : vary mixing during lo [bool]
    n_l          : # polynomes Legendre  [int]
    n_iw         : # Matsubara freq      [int]
    n_tau        : # imaginary time      [int]

    n_cycles     : # of measures         [int]
    n_warmup     : warmup for therma.    [int]
    length_cycle : MC steps              [int]
    n_loops      : # boucles DMFT        [int]
    threshold    : arrêt DMFT si         [float]
                   erreur atteint sur les fct de Green

    fit          : fit Sigma or no       [bool]
    fit_min_n    : left window for       [integer]
                    fiting Sigma_iw
    fit_max_n    : right window for      [integer]
                    fiting Sigma_iw

   file_name    : name of the output file   [str]
   folder       : folder of output file     [str]

   file_app     : restart a DMFT loop from  [bool]
                 "DMFT_data/folder/file.h5"
    ----------
    """

    # parameters for cthyb solve
    if fit :
        p = {"n_cycles":n_cycles, "n_warmup_cycles":n_warmup, "length_cycle":length_cycles, "random_seed":2132 * mpi.rank + 121, \
         "measure_density_matrix" : measure_density, "use_norm_as_weight" : measure_density, \
         "perform_tail_fit":fit, "fit_max_moment":4, "fit_min_n":fit_min_n, "fit_max_n":fit_max_n}
    else :
        p = {"n_cycles":n_cycles, "n_warmup_cycles":n_warmup, "length_cycle":length_cycles, "random_seed":2132 * mpi.rank + 121, \
         "measure_density_matrix" : measure_density, "use_norm_as_weight" : measure_density, \
         "perform_tail_fit":fit}
    # creation of h5
    if not file_app :
        it = -1      # new DMFT calculation --> start at iteration 0
        if mpi.is_master_node():
            print("------------------------------------------------\nDMFT Calculation from loop 0.\n")

            # if folder doesn't exist, create it.
            if not os.path.isdir("DMFT_data/%s"%folder):
                print("\nFolder does not exsit. Creation of the folder %s\n"%folder)
                os.mkdir("DMFT_data/%s"%folder)

            # change the name file if it already exists in the folder
            file_name_ = file_name
            if os.path.exists("DMFT_data/%s/%s.h5"%(folder, file_name)):
                print(f"The file 'DMFT_data/{folder}/{file_name}.h5' already exists. "
                    f"New name of the file: 'DMFT_data/{folder}/{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5'.")
                file_name_ += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

            # fill h5 with dmft parameters + create pandas df to print convergence parameters
            with HDFArchive("DMFT_data/%s/%s.h5"%(folder, file_name_), "w") as A:
                A.create_group("parameters")
                A["parameters"]["loop_1"] = {"beta" : beta, "U" : U, "J" : J, "mu_dc" : mu_dc, "n_iw" : n_iw, "n_tau" : n_tau, "n_max_loops" : n_loops, "mix" : mix,  **p}

                # panda array for the global convergence informations
                columns = ["μ", "Gt2g(b/2)", "Geg(b/2)", "n_t2gG", "n_egG", "n_totG", "hf_t2g", "hf_eg", "ΔG/G", "av._sign", "av._order", "ACT", "mix", "time", "full_time"]
                df = pa.DataFrame(columns=columns)

    else :
        # if the file already exists, load the parameters
        with HDFArchive("DMFT_data/%s/%s.h5"%(folder, file_name), "r") as A:
            it = len(A) - 2
            Sigma_init   = A[f"iteration_{it}"]["Sigma_iw"]
            Sigma_iw_old = Sigma_init.copy()
            Giw_init     = A[f"iteration_{it}"]["G_iw"]
            Giw_old      = Giw_init.copy()

            mu_init      = A[f"iteration_{it}"]["mu"]
            dc_imp       = A[f"iteration_{it}"]["dc_imp"]
            dc_energ     = A[f"iteration_{it}"]["dc_energ"]

            beta, U, n_iw, n_tau = (A["parameters"][f"loop_{len(A['parameters'])}"]["beta"], A["parameters"][f"loop_{len(A['parameters'])}"]["U"], \
                                    A["parameters"][f"loop_{len(A['parameters'])}"]["n_iw"], A["parameters"][f"loop_{len(A['parameters'])}"]["n_tau"])
        time.sleep(1)
        if mpi.is_master_node():
            print("------------------------------------------------\nDMFT Calculation from file %s"%(folder + file_name))
            print("!!! Took the parameters U=%.2f, beta=%.2f, n_iw=%d, n_tau=%d, mu_init=%.4f, dc_imp from the latter file !!!\n"%(U, beta, n_iw, n_tau, mu_init))
            with HDFArchive("DMFT_data/%s/%s.h5"%(folder, file_name), "a") as A:
                A["parameters"][f"loop_{len(A['parameters'])+1}"] = {"beta" : beta, "U" : U, "J" : J, "mu_dc" : mu_dc, "n_iw" : n_iw, "n_tau" : n_tau, "n_max_loops" : n_loops, "mix_init" : mix,  **p}
            df = pa.read_csv("DMFT_data/%s/convergence_%s.txt" % (folder, file_name), sep="|", index_col=0)
            print("Reading file...\n", df, "\n")
            file_name_ = file_name

    # initialization of SumkDFT
    SK = SumkDFT(hdf_file='w90_to_triqs.h5', beta = beta, n_iw = n_iw, use_dft_blocks=True)

    # Green's function structure
    gf_struct  = list(SK.gf_struct_solver[0].items())
    n_orbitals = SK.corr_shells[0]['dim'] # number of orbtials

     # initializing solver and interacting hamiltonian
    rot_basis = spherical_to_cubic(2, convention='wannier90')
    Uijkl = U_matrix_slater(2, U_int=U, J_hund=J, basis='other', T=rot_basis)
    Umat, Upmat = reduce_4index_to_2index(Uijkl)
    h_int = h_int_density(["up", "down"], n_orbitals, U=Umat, Uprime=Upmat, map_operator_structure=SK.sumk_to_solver[0])

    print(h_int)
    print(print("\n\n", SK.eff_atomic_levels(), "\n\n"))

    S = Solver(beta = beta, n_iw = n_iw, n_tau = n_tau, gf_struct = gf_struct)

    # initialization of Sigma
    S.Sigma_iw.zero()

    if not file_app:
        if mpi.is_master_node(): print("\nNo initialization on Sigma. Sigma set to zero.\n")
        Sigma_iw_old = S.Sigma_iw.copy()
        Sigma_iw_old.zero()
        Giw_old = S.Sigma_iw.copy()
        Giw_old.zero()

    else :
        if mpi.is_master_node(): print("\nSigma initialize.")
        S.Sigma_iw << Sigma_init
        S.G_iw << Giw_init
        SK.set_dc(dc_imp, dc_energ) # double counting
    
    # set chemical potential
    SK.set_mu(mu_init)
    mu = mu_init

    # DMFT loop with self-consistency
    i = it+1
    break_var = False
    while i < it+n_loops+1:

        start = time.time()

        # calculation of the chemical potential
        SK.symm_deg_gf(S.Sigma_iw, ish=0)
        SK.set_Sigma([S.Sigma_iw])
        mu = SK.calc_mu(precision=1e-2)

        # Green's function of the lattice
        S.G_iw << SK.extract_G_loc()[0]
        G_lat = S.G_iw.copy()

        if i==0 :
            dm = S.G_iw.density()
            SK.calc_dc(dm, orb=0, use_dc_value=mu_dc)
            S.Sigma_iw << SK.dc_imp[0]['up'][0,0]

        # Weiss's field calculation
        S.G0_iw << dyson(Sigma_iw = S.Sigma_iw, G_iw = S.G_iw)

        # Solve the impurity problem
        S.solve(h_int = h_int, **p)

        # mixing
        if i>0:
            if mpi.is_master_node(): print("MIX")
            S.G_iw     << mix*S.G_iw + (1-mix)*Giw_old
            S.Sigma_iw << mix*S.Sigma_iw + (1-mix)*Sigma_iw_old
        Giw_old = S.G_iw.copy()
        Sigma_iw_old = S.Sigma_iw.copy()

        ### Double-counting correction
        dm = S.G_iw.density()
        SK.calc_dc(dm, orb=0, use_dc_value=mu_dc)

        # save
        end = time.time()

        Glat_minus_Gimp = 0
        for j in range(n_orbitals):
            Glat_minus_Gimp +=  np.sqrt(np.sum( np.abs(S.G_iw['up_{}'.format(j)].data - G_lat['up_{}'.format(j)].data)**2 + \
                                np.abs(S.G_iw['down_{}'.format(j)].data - G_lat['down_{}'.format(j)].data)**2 )) / \
                                np.sqrt(np.sum(np.abs(S.G_iw['up_{}'.format(j)].data)**2 + np.abs(S.G_iw['down_{}'.format(j)].data)**2 ))

        if mpi.is_master_node():
                
                Gb2_t2g = S.G_tau["up_0"].data.real.flatten()[int(n_tau/2)]
                Gb2_eg = S.G_tau["up_3"].data.real.flatten()[int(n_tau/2)]

                print("\n\nIteration = %i / %i Finished." % (i+1, it+1+n_loops))
                with HDFArchive("DMFT_data/%s/%s.h5"%(folder, file_name_), "a") as A:
                    A.create_group(f"iteration_{i}")
                    A[f"iteration_{i}"]["G_iw"] = S.G_iw
                    A[f"iteration_{i}"]["G_tau"] = S.G_tau
                    A[f"iteration_{i}"]["G0_iw"] = S.G0_iw
                    A[f"iteration_{i}"]["Sigma_iw"] = S.Sigma_iw
                    A[f"iteration_{i}"]["Delta_tau"] = S.Delta_tau

                    A[f"iteration_{i}"]["mu"] = mu
                    A[f"iteration_{i}"]['dc_imp'] = SK.dc_imp
                    A[f"iteration_{i}"]['dc_energ'] = SK.dc_energ

                    A[f"iteration_{i}"]["density_from_giw"] = S.G_iw.density()
                    A[f"iteration_{i}"]["Z"] = [1 / (1 - ((S.Sigma_iw['up_%d'%j](0).imag)[0,0] * beta / (np.pi))) for j in range(n_orbitals)]
                    A[f"iteration_{i}"]["Gb2"] = [S.G_tau["up_%d"%j].data.real.flatten()[int(n_tau/2)] for j in range(n_orbitals)]

                    A[f"iteration_{i}"]["auto_corr_time"] = S.auto_corr_time
                    A[f"iteration_{i}"]["average_order"] = S.average_order
                    A[f"iteration_{i}"]["average_sign"] = S.average_sign

                    A[f"iteration_{i}"]["mix"] = mix
                    A[f"iteration_{i}"]["time"] = end - start

                    if i == 0 :
                        A[f"iteration_{i}"]["full_time"] = end - start
                    else :
                        A[f"iteration_{i}"]["full_time"] = end - start + A[f"iteration_{i-1}"]["full_time"]

                    df.loc[len(df)] = [mu, Gb2_t2g, Gb2_eg, A[f"iteration_{i}"]["density_from_giw"]["up_0"][0,0].real, A[f"iteration_{i}"]["density_from_giw"]["up_3"][0,0].real,\
                                       S.G_iw.total_density().real, S.Sigma_moments["up_0"][0,0,0].real, S.Sigma_moments["up_3"][0,0,0].real, \
                                       Glat_minus_Gimp, S.average_sign, S.average_order, S.auto_corr_time, mix, end - start, A[f"iteration_{i}"]["full_time"]]

                    if measure_density :
                        A[f"iteration_{i}"]["density_matrix"] = S.density_matrix
                        A[f"iteration_{i}"]["h_loc_diagonalization"] = S.h_loc_diagonalization

                print("\n--------------------------------------------\n")
                print(df)
                df.to_csv("DMFT_data/%s/convergence_%s.txt"%(folder, file_name_), sep="|", float_format="%.5f")

        if Glat_minus_Gimp < threshold and i > -1:
            if mpi.is_master_node():
                print("Stopping DMFT loops : threshold reach at ", i+1, "loops.")
                print("threshold = ", threshold, " ; err = ", Glat_minus_Gimp)
            break

        elif (i == (n_loops - 1)) and mix_vary:
            if mpi.is_master_node():
                print("\n NO CONVERGENCE, change mix = %.2f to mix = %.2f"%(mix, mix - 0.2))
            mix -= 0.2
            n_loops += 8
            if mix < 0.09 :
                if mpi.is_master_node():
                    print("mix < 0.3 : NO CONVERGENCE. ABORT OF THE DMFT LOOP.")
                break_var = True
                break

        i += 1

    if mpi.is_master_node():

        print("\nDMFT ---> Done !")

        print("\n Convergence :\n")
        print(df)
        df.to_csv("DMFT_data/%s/convergence_%s.txt"%(folder, file_name_), sep="|", float_format="%.5f")

        print("\nFiles : DMFT_data/%s/%s.h5"%(folder, file_name_))
        print("      : DMFT_data/%s/convergence_%s.txt"%(folder, file_name_))
        print("------------------------------------------------")

    return S.Sigma_iw.copy(), S.G_iw.copy(), break_var, Glat_minus_Gimp


#############################################################
####################   Variables   ##########################
#############################################################

######## system parameters #######
beta = 40.           # eV-1
U = 7.0             # eV
J = 1.1             # eV
mu_init = -1.150475        # eV
mu_dc   = 52.        # eV

n_l   = 30
n_iw  = 1024
n_tau = 10001

######## MC & DMFT loop #######
n_cycles         = 10000                      # 50000
length_cycle     = 300                     # 1000
n_warmup         = 5000     # 400000
n_loops          = 1
threshold        = 1e-3

mix_init = 0.6
mix_vary = False

measure_density = True

######## fitting #######
fit       = False
fit_min_w = 8.0    # eV
fit_max_w = 15.0    # eV

######## save #######
folder    = "test_tail"
file_name = "DMFT"

########  initialisation  ###########
file_app = True    # continuer un calcul

###########################################################################################
##############################           Main            ##################################
###########################################################################################

fit_min_n = int(1/2 * (beta*fit_min_w/np.pi - 1))
fit_max_n = int(1/2 * (beta*fit_max_w/np.pi - 1))

break_var = False

_, _, break_var, Glat_minus_Gimp = DMFT(beta, U, J, mix_init, mix_vary, n_l, n_iw, n_tau, \
    n_cycles, int(n_warmup), length_cycle, n_loops, threshold,\
    fit, fit_min_n, fit_max_n,
        file_name + "_U%.2f_beta%.2f"%(U, beta), folder, measure_density, \
        file_app, mu_init, mu_dc)

if break_var == True :
    if mpi.is_master_node() : print("End of the calculation. NO CONVERGENCE REACHED.")
