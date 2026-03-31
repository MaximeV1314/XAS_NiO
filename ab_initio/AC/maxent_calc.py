import matplotlib.pyplot as plt

from triqs.gf import *
from triqs.operators import *
from triqs_maxent import *
from triqs_dft_tools.sumk_dft import *

from h5 import *
import os

###############################################################################
#####################            Fonctions           ##########################
###############################################################################

def plot(res_n, folder, U, beta, name):

    fig1 = plt.figure()
    # chi2(alpha) and linefit
    plt.subplot(2, 2, 1)
    res_n.analyzer_results['LineFitAnalyzer'].plot_linefit()
    res_n.plot_chi2()

    # curvature(alpha)
    plt.subplot(2, 2, 3)
    res_n.analyzer_results['Chi2CurvatureAnalyzer'].plot_curvature()

    # probablity(alpha)
    plt.subplot(2, 2, 2)
    res_n.plot_probability()

    # backtransformed G_rec(tau) and original G(tau)
    # by default (plot_G=True) also original G(tau) is plotted
    plt.subplot(2, 2, 4)
    res_n.plot_G_rec(alpha_index=n_alpha-1)

    plt.tight_layout()
    plt.savefig(folder + "results_Sigma_U%.2f_beta%.2f_orb%s.png"%(U,beta, name), dpi = 150)

    plt.close(fig1)

def maxent_minimization_ISC(S_iw, dc, err, alpha_min, alpha_max, n_alpha, n_w, np_omega, U, beta, folder):

    print("error = %.3f"%err, " , alpha_min = ", alpha_min, " , alpha_max = ", alpha_max)

    isc = InversionSigmaContinuator(S_iw, dc)
    res = {}
    for orb, gaux_iw in isc.Gaux_iw:

        tm = TauMaxEnt(cost_function='bryan', probability='normal')
        tm.alpha_mesh = LogAlphaMesh(alpha_min=alpha_min, alpha_max=alpha_max, n_points=n_alpha)
        tm.set_G_iw(gaux_iw)
        tm.omega = HyperbolicOmegaMesh(omega_min=-20, omega_max=20, n_points=n_w)
        tm.set_error(err)
        res[orb] = tm.run()

        plot(res[orb], folder, U, beta, orb+"ISC")

    name_list   = [orb for orb in res]
    Aaux_w_gf = [Gf(mesh = MeshReFreq(w_min=-20, w_max=20, n_w=n_w), data=res[orb].get_A_out()) for orb in res]
    Aaux_w_gf = BlockGf(name_list=name_list, block_list=Aaux_w_gf, make_copies=False)

    isc.set_Gaux_w_from_Aaux_w({orb:res[orb].get_A_out() for orb in res}, res["up_0"].omega, np_interp_A=10000, np_omega=np_omega, w_min=-20., w_max=20., broadening_factor=1.)
    Sigma_w = BlockGf(name_list=name_list, block_list=[isc.S_w[orb] for orb in res], make_copies=False)

    with HDFArchive(folder + 'maxent_U%.2f_beta%.2f_Sigma_w_ISC.h5'%(U, beta),'w') as ar:
        ar["Aaux_w"] = Aaux_w_gf
        ar["Sigma_w"] = Sigma_w

    return Sigma_w

def maxent_minimization_DSC(S_iw, err, alpha_min, alpha_max, n_alpha, n_w, np_omega, U, beta, folder):

    print("error = %.3f"%err, " , alpha_min = ", alpha_min, " , alpha_max = ", alpha_max)

    dsc = DirectSigmaContinuator(S_iw)
    res = {}
    for orb, gaux_iw in dsc.Gaux_iw:

        tm = TauMaxEnt(cost_function='bryan', probability='normal')
        tm.alpha_mesh = LogAlphaMesh(alpha_min=alpha_min, alpha_max=alpha_max, n_points=n_alpha)
        tm.set_G_iw(gaux_iw)
        tm.omega = HyperbolicOmegaMesh(omega_min=-20, omega_max=20, n_points=n_w)
        tm.set_error(err)
        res[orb] = tm.run()

        plot(res[orb], folder, U, beta, orb+"DSC")

    name_list   = [orb for orb in res]
    Aaux_w_gf = [Gf(mesh = MeshReFreq(w_min=-20, w_max=20, n_w=n_w), data=res[orb].get_A_out()) for orb in res]
    Aaux_w_gf = BlockGf(name_list=name_list, block_list=Aaux_w_gf, make_copies=False)

    dsc.set_Gaux_w_from_Aaux_w({orb:res[orb].get_A_out() for orb in res}, res["up_0"].omega, np_interp_A=10000, np_omega=np_omega, w_min=-20., w_max=20., broadening_factor=1.)
    Sigma_w = BlockGf(name_list=name_list, block_list=[dsc.S_w[orb] for orb in res], make_copies=False)

    with HDFArchive(folder + 'maxent_U%.2f_beta%.2f_Sigma_w_DSC.h5'%(U, beta),'w') as ar:
        ar["Aaux_w"] = Aaux_w_gf
        ar["Sigma_w"] = Sigma_w

    return Sigma_w

def G_continuation(Sigma_w, dc_imp, dc_energ, mu, n_iw, path):

    SK = SumkDFT(hdf_file=path+'w90_to_triqs.h5', mesh = Sigma_w.mesh, beta = beta, n_iw = n_iw, use_dft_blocks=True)
    SK.set_dc(dc_imp, dc_energ)
    SK.set_mu(mu)
    SK.set_Sigma([Sigma_w])
    G_w = SK.extract_G_loc()[0]

    with HDFArchive(folder + 'maxent_U7.00_beta40.00_G_w.h5','w') as ar:
        ar["G_w"] = G_w

###############################################################################
#####################            Variables           ##########################
###############################################################################

# maxent parameters
error_se = 0.01

alpha_min_se = 1e-6
alpha_max_se = 1e-1

n_alpha = 40
n_w = 500
np_omega = 1000

#system parameters
U = 7.0
beta = 40.

# load and save
path = "../DMFT_NiO_CTSEG/DMFT_data/results/"
# path = "../../DMFT/DMFT_NiO_CTSEG/DMFT_data/results/"
file = "DMFT_U%.2f_beta%.2f.h5"%(U, beta)

###############################################################################
#####################              main              ##########################
###############################################################################

folder = "MaxEnt_data/"
with HDFArchive(path + file, "r") as A :
    it = len(A) - 2
    S_iw  = A["iteration_%d"%it]["Sigma_iw"]
    mu    = A["iteration_%d"%it]["mu"]
    n_iw  = A["parameters"]["loop_1"]["n_iw"]
    dc_energ = A["iteration_%d"%it]["dc_energ"]
    dc_imp   = A["iteration_%d"%it]["dc_imp"]

print("\n------------  Minimisation of Sigma  ISC  ------------\n")
Sigma_w = maxent_minimization_ISC(S_iw, dc_imp[0]["up"][0,0], error_se, alpha_min_se, alpha_max_se, n_alpha, n_w, np_omega, U, beta, folder)

#print("\n------------  Minimisation of Sigma  DSC  ------------\n")
#Sigma_w = maxent_minimization_DSC(S_iw, error_se, alpha_min_se, alpha_max_se, n_alpha, n_w, np_omega, U, beta, folder)

with HDFArchive(folder + "maxent_U7.00_beta40.00_Sigma_w_ISC.h5", "r") as A :
    Sigma_w = A["Sigma_w"]

G_continuation(Sigma_w, dc_imp, dc_energ, mu, n_iw, "../DMFT_NiO_CTSEG/")
