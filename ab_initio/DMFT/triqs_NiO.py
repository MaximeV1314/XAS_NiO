from triqs_dft_tools.sumk_dft import *
from triqs.gf import *
from triqs.operators import *
from triqs.operators.util import *
from triqs.atom_diag import *
from h5 import *

from triqs_cthyb import Solver
from triqs_cthyb.tail_fit import tail_fit
import sys

import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np
from scipy import interpolate
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

def comm(A,B): return A*B - B*A
def anticomm(A,B): return A*B + B*A

def compute_sigma_hf(h_int, rho):
    """
    Computes the Hartree-Fock part of the self-energy
    using the density

    Parameters :
    ------------
        - rho : dict of matrices
            The density matrix
    """

    sigma_hf = {k:np.zeros_like(m) for k, m in rho.items()}

    for ((_,(s1,o1)),(_,(s2,o2)),(_,(s3,o3)),(_,(s4,o4))), coef in h_int:
        if s1 == s4 and s2 == s3:
            sigma_hf[s1][o1,o4] += coef * rho[s2][o2,o3]
            sigma_hf[s2][o2,o3] += coef * rho[s1][o1,o4]

    return sigma_hf

def DMFT(beta : float, U : float, mix : float, mix_vary:bool, n_l : int, n_iw : int, n_tau : int, \
         n_cycles : int, n_warmup : int, length_cycles : int, n_loops : int, threshold : float,\
         fit : bool, fit_min_w : float, fit_max_w : float, \
         file_name : str, folder : str = "", measure_density : bool = False, \
            Sigma_init = "none", Giw_init = "none", file_app = False, mu_init = "none"):

    """
    DMFT loop
    Parameters
    ----------
    beta         : temperature           [float]
    U            : interaction term      [float]
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
    fit_min_w    : left window for       [float]
                    fiting Sigma_iw
    fit_max_w    : right window for      [float]
                    fiting Sigma_iw

   file_name    : name of the output file   [str]
   folder       : folder of output file     [str]

   Sigma_init   : initialize self energy    [triqs Gf]
   file_app     : restart a DMFT loop from  [bool]
                 "DMFT_data/folder/file.h5"
    ----------
    """

    # parameters for cthyb solve
    p = {"n_cycles":n_cycles, "n_warmup_cycles":n_warmup, "length_cycle":length_cycles, "random_seed":2132 * mpi.rank + 121, "measure_G_l":False, \
         "measure_density_matrix" : measure_density, "use_norm_as_weight" : measure_density, "perform_post_proc" : False}
    # creation of h5
    if not file_app :
        it = -1      # new DMFT calculation --> start at iteration 0
        if mpi.is_master_node():
            print("------------------------------------------------\nDMFT Calculation from loop 0.\n")

            if not os.path.isdir("DMFT_data/%s"%folder):
                print("\nFolder does not exsit. Creation of the folder %s\n"%folder)
                os.mkdir("DMFT_data/%s"%folder)

            file_name_ = file_name
            if os.path.exists("DMFT_data/%s/%s.h5"%(folder, file_name)):
                print(f"The file 'DMFT_data/{folder}/{file_name}.h5' already exists. "
                    f"New name of the file: 'DMFT_data/{folder}/{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5'.")
                file_name_ += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        
            with HDFArchive("DMFT_data/%s/%s.h5"%(folder, file_name_), "w") as A:
                A.create_group("parameters")
                A["parameters"]["loop_1"] = {"beta" : beta, "U" : U, "n_iw" : n_iw, "n_tau" : n_tau, "fit_min_w" : fit_min_w, "fit_max_w" : fit_max_w, "n_max_loops" : n_loops, "mix" : mix,  **p}

                # panda array for the global convergence informations
                columns = ["μ", "Z_up", "Z_up", "Geg(b/2)", "Gt2g(b/2)", "n_eg", "n_t2g", "Δμ", "ΔG/G", "av._sign", "av._order", "ACT", "mix", "time", "full_time"]
                df = pa.DataFrame(columns=columns)
    
    else :
        with HDFArchive("DMFT_data/%s/%s.h5"%(folder, file_name), "r") as A:
            it = len(A) - 2
            Sigma_init   = A[f"iteration_{it}"]["Sigma_iw"]
            Sigma_iw_old = Sigma_init.copy()
            Giw_old      = A[f"iteration_{it}"]["G_iw"]
            beta, U, n_iw, n_tau = (A["parameters"][f"loop_{len(A['parameters'])}"]["beta"], A["parameters"][f"loop_{len(A['parameters'])}"]["U"], \
                                    A["parameters"][f"loop_{len(A['parameters'])}"]["n_iw"], A["parameters"][f"loop_{len(A['parameters'])}"]["n_tau"])
        time.sleep(1)
        if mpi.is_master_node():
            print("------------------------------------------------\nDMFT Calculation from file %s"%(folder + file_name))
            print("!!! Took the parameters U=%.2f, beta=%.2f, n_iw=%d and n_tau=%d from the latter file !!!\n"%(U, beta, n_iw, n_tau))
            with HDFArchive("DMFT_data/%s/%s.h5"%(folder, file_name), "a") as A:
                A["parameters"][f"loop_{len(A['parameters'])+1}"] = {"beta" : beta, "U" : U, "n_iw" : n_iw, "n_tau" : n_tau, "fit_min_w" : fit_min_w, "fit_max_w" : fit_max_w, "n_max_loops" : n_loops, "mix_init" : mix,  **p}
            df = pa.read_csv("DMFT_data/%s/convergence_%s.txt" % (folder, file_name), sep="\t", index_col=0)
            print("Reading file...\n", df, "\n")
            file_name_ = file_name

    # initialization of SumkDFT
    SK = SumkDFT(hdf_file='w90_to_triqs.h5', beta = beta, n_iw = n_iw)

    # DMFT loop with self-consistency
    gf_struct = SK.gf_struct_solver[0]
    print(gf_struct)

    S = Solver(beta = beta, n_iw = n_iw, n_tau = n_tau, n_l = n_l, gf_struct = gf_struct)
    h_int = U * (n('up_0',0) * n('down_0',0) + n('up_0',1) * n('down_0',1) + n('up_0',2) * n('down_0',2) + \
                 n('up_0',3) * n('down_0',3) + n('up_0',4) * n('down_0',4))


    # initialization of Sigma
    S.Sigma_iw.zero()
    # S.Sigma_iw["up_0"] << 0.5*U + U*U/4*inverse(iOmega_n-U/2)
    # S.Sigma_iw["down_0"] << 0.5*U + U*U/4*inverse(iOmega_n-U/2)
    if type(Sigma_init) != type(S.Sigma_iw):
        if mpi.is_master_node(): print("\nNo initialization on Sigma. Sigma set to zero.\n")
        Sigma_iw_old = S.Sigma_iw.copy()
        Sigma_iw_old.zero()
        Giw_old = S.Sigma_iw.copy()
        Giw_old.zero()

    else :
        if mpi.is_master_node(): print("\nSigma initialize.")
        S.Sigma_iw << Sigma_init


        if not file_app:
            Sigma_iw_old = S.Sigma_iw.copy()
            if type(Giw_init) != type(Sigma_iw_old):
                if mpi.is_master_node(): print("\nNo initialization on Giw. Giw set to zero.\n")
                Giw_old = Sigma_iw_old.copy()
                Giw_old.zero()
            else:
                if mpi.is_master_node(): print("\n Initialization on Giw.\n")
                Giw_old = Giw_init.copy()

    i = it+1
    break_var = False
    while i < it+n_loops+1:
        
        start = time.time()

        #symmetrisation
        g = (S.G_iw["up_0"] + S.G_iw["down_0"])/2
        S.G_iw["up_0"] << g
        S.G_iw["down_0"] << g

        SK.set_Sigma([S.Sigma_iw])
        if mu_init == "none" :
            mu = SK.calc_mu(precision=1e-6)
        else :
           mu = mu_init
           SK.set_mu(mu_init)

        # Green's function of the lattice
        S.G_iw << SK.extract_G_loc()[0]
        G_lat = S.G_iw.copy()

        # Weiss's field calculation
        S.G0_iw << dyson(Sigma_iw = S.Sigma_iw, G_iw = S.G_iw)

        # Solve the impurity problem
        S.solve(h_int = h_int, **p)

        # fitting by calculating the hatree self-energy
        G_iw = S.G0_iw.copy()
        for bl, g in S.G_tau:
            bl_size = g.target_shape[0]
            known_moments = make_zero_tail(g, 4)
            known_moments[1,...] = np.eye(bl_size)
            G_iw[bl].set_from_fourier(g, known_moments)

        Sigma_iw = dyson(G0_iw=S.G0_iw, G_iw=G_iw)
        Sigma_iw_nofit = Sigma_iw.copy()

        rho = {name:np.eye(g.target_shape[0])+g(0) for name, g in S.G_tau}
        if fit :
            hf = compute_sigma_hf(h_int, rho)
            for name, sigma in Sigma_iw:
                sigma << make_hermitian(sigma)
                km = np.array([hf[name]])
                tail, _ = fit_hermitian_tail_on_window(
                        sigma,
                        n_min = int(1/2 * (beta*fit_min_w/np.pi - 1)),
                        n_max = int(1/2 * (beta*fit_max_w/np.pi - 1)),
                        known_moments = km,
                        n_tail_max = 10*len(sigma.mesh),
                        expansion_order = 5
                        )
                replace_by_tail(sigma, tail, n_min=int(1/2 * (beta*fit_min_w/np.pi - 1)))

        S.G_iw << dyson(G0_iw=S.G0_iw, Sigma_iw=Sigma_iw)
        S.Sigma_iw << Sigma_iw

        # mixing
        if not (i==0 and (type(Sigma_init) != type(S.Sigma_iw))):
            if mpi.is_master_node(): print("MIX")
            S.G_iw     << mix*S.G_iw + (1-mix)*Giw_old
            S.Sigma_iw << mix*S.Sigma_iw + (1-mix)*Sigma_iw_old
        Giw_old = S.G_iw.copy()
        Sigma_iw_old = S.Sigma_iw.copy()

        # save
        end = time.time()

        Glat_minus_Gimp =  np.sqrt(np.sum( np.abs(S.G_iw["up_0"].data - G_lat["up_0"].data)**2 + \
                               np.abs(S.G_iw["down_0"].data - G_lat["down_0"].data)**2 ))
        Glat_minus_Gimp /= np.sqrt(np.sum(np.abs(S.G_iw["up_0"].data)**2 + np.abs(S.G_iw["down_0"].data)**2 ))

        if mpi.is_master_node():
                
                Gb2_eg = S.G_tau["up_0"][0,0].data.real.flatten()[int(n_tau/2)]
                Gb2_t2g = S.G_tau["up_0"][3,3].data.real.flatten()[int(n_tau/2)]
                print("\n\nIteration = %i / %i Finished." % (i+1, it+1+n_loops))
                with HDFArchive("DMFT_data/%s/%s.h5"%(folder, file_name_), "a") as A:
                    A.create_group(f"iteration_{i}")
                    A[f"iteration_{i}"]["G_iw"] = S.G_iw
                    A[f"iteration_{i}"]["G_tau"] = S.G_tau
                    A[f"iteration_{i}"]["G0_iw"] = S.G0_iw
                    A[f"iteration_{i}"]["Sigma_iw"] = S.Sigma_iw
                    A[f"iteration_{i}"]["Sigma_iw_nofit"] = Sigma_iw_nofit
                    A[f"iteration_{i}"]["Delta_tau"] = S.Delta_tau
                    A[f"iteration_{i}"]["mu"] = mu
                    A[f"iteration_{i}"]["density_from_gtau"] = rho
                    A[f"iteration_{i}"]["density_from_giw_up"] = [S.G_iw["up_0"][i,i].density().real for i in range(5)]
                    A[f"iteration_{i}"]["density_from_giw_down"] = [S.G_iw["down_0"][i,i].density().real for i in range(5)]
                    A[f"iteration_{i}"]["Z_up"] = [1 / (1 - ((S.Sigma_iw['up_0'](0)[i,i].imag) * beta / (np.pi))) for i in range(5)]
                    A[f"iteration_{i}"]["Z_down"] = [1 / (1 - ((S.Sigma_iw['down_0'](0)[i,i].imag) * beta / (np.pi))) for i in range(5)]
                    A[f"iteration_{i}"]["Gb2_up"] = [S.G_tau["up_0"][i,i].data.real.flatten()[int(n_tau/2)] for i in range(5)]
                    A[f"iteration_{i}"]["Gb2_down"] = [S.G_tau["down_0"][i,i].data.real.flatten()[int(n_tau/2)] for i in range(5)]
                    A[f"iteration_{i}"]["auto_corr_time"] = S.auto_corr_time
                    A[f"iteration_{i}"]["average_order"] = S.average_order
                    A[f"iteration_{i}"]["average_sign"] = S.average_sign
                    A[f"iteration_{i}"]["mix"] = mix

                    A[f"iteration_{i}"]["time"] = end - start

                    if i == 0 :
                        A[f"iteration_{i}"]["full_time"] = end - start

                        print(S.G_iw["up_0"][0,0].density().real)
                        print(S.G_iw["up_0"][3,3].density().real)

                        df.loc[len(df)] = [mu, A[f"iteration_{i}"]["Z_up"][0], A[f"iteration_{i}"]["Z_up"][3], Gb2_eg, Gb2_t2g, \
                                           A[f"iteration_{i}"]["density_from_giw_up"][0], A[f"iteration_{i}"]["density_from_giw_up"][3], 0,\
                                           Glat_minus_Gimp, S.average_sign, S.average_order, S.auto_corr_time, mix, end - start, end - start]

                    else :
                        A[f"iteration_{i}"]["full_time"] = end - start + A[f"iteration_{i-1}"]["full_time"]
                            
                        delta_mu = mu - A[f"iteration_{i-1}"]["mu"]    
                        df.loc[len(df)] = [mu, A[f"iteration_{i}"]["Z_up"][0], A[f"iteration_{i}"]["Z_up"][3], Gb2_eg, Gb2_t2g, \
                                            A[f"iteration_{i}"]["density_from_giw_up"][0], A[f"iteration_{i}"]["density_from_giw_up"][3], delta_mu,\
                                            Glat_minus_Gimp, S.average_sign, S.average_order, S.auto_corr_time, mix, end - start, end - start + A[f"iteration_{i-1}"]["full_time"]]

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
beta = 40           # eV-1
U = 7.1             # eV
mu_init = "none"    # eV

n_l   = 30
n_iw  = 1024
n_tau = 10001

######## MC & DMFT loop #######
n_cycles         = 2000                         # 20000
length_cycle     = 500
n_warmup         = 1000000/length_cycle     # 5000000
n_loops          = 2
threshold        = -1

mix_init = 0.8
mix_vary = False

######## fitting #######
fit       = False
fit_min_w = 1.25    # eV
fit_max_w = 2.50    # eV

######## save #######
folder    = "test"
file_name = "DMFT"

########  initialisation  ###########
file_app = False    # continuer un calcul

Sigma_init = ""
G_init = ""
measure_density = False

"""
with HDFArchive("DMFT_data/hist/beta40.00/DMFT_U2.30_beta40.00.h5", "r") as A:
    it = len(A) - 2
    Sigma_init = A["iteration_%d"%it]["Sigma_iw"]
    G_init = A["iteration_%d"%it]["G_iw"]
"""

###########################################################################################
##############################           Main            ##################################
###########################################################################################

break_var = False

_, _, break_var, Glat_minus_Gimp = DMFT(beta, U, mix_init, mix_vary, n_l, n_iw, n_tau, \
    n_cycles, int(n_warmup), length_cycle, n_loops, threshold,\
    fit, fit_min_w, fit_max_w,
        file_name + "_U%.2f_beta%.2f"%(U, beta), folder, measure_density, \
            Sigma_init, G_init, file_app, mu_init)

if break_var == True :
    if mpi.is_master_node() : print("End of the calculation. NO CONVERGENCE REACHED.")
