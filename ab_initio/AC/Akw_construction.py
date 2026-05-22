import numpy as np
import matplotlib.pyplot as plt

from triqs.gf import *
from triqs.gf.tools import *
from triqs.operators import *
from triqs_dft_tools.sumk_dft import *
from triqs_dft_tools.sumk_dft_tools import *



#-----------------------------------------------------------------
#                       DMFT data loading
#-----------------------------------------------------------------

def k_path_extraction(kgrid):

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

DMFT_folder = "../../DMFT/DMFT_NiO_CTSEG/"
Nk=1000

with HDFArchive("MaxEnt_data/maxent_U7.00_beta40.00_Sigma_w_ISC.h5", "r") as A :
    Sigma_w = A["Sigma_w"]

with HDFArchive("MaxEnt_data/maxent_U7.00_beta40.00_G_w.h5", "r") as A :
    G_w = A["G_w"]

with HDFArchive(DMFT_folder+"DMFT_data/results/DMFT_U7.00_beta40.00.h5", "r") as A :
    it = len(A) - 2
    mu    = A["iteration_%d"%it]["mu"]
    dc_energ = A["iteration_%d"%it]["dc_energ"]
    dc_imp   = A["iteration_%d"%it]["dc_imp"]

with HDFArchive(DMFT_folder+'w90_to_triqs.h5', "r") as A :
    kpts = A["dft_input"]["kpts"]

kpath_indices, kpath_points = k_path_extraction(kpts)
omega = np.array([w  for w in Sigma_w.mesh.values()])

SK = SumkDFTTools(hdf_file = DMFT_folder + 'w90_to_triqs.h5', mesh = Sigma_w.mesh, beta = 40., use_dft_blocks=True)

SK.set_Sigma([Sigma_w])
SK.set_mu(mu)
SK.set_dc(dc_imp,dc_energ)



A_k = np.zeros((len(kpath_indices), len(omega)))
for i, ik in enumerate(kpath_indices):
    G_ki = SK.lattice_gf(ik=ik)
    for j in range(8):
        A_k[i] += G_ki["up"][j,j].data.imag

A_k *= -2/np.pi

plt.ylim(300, 600)
plt.imshow(A_k.T, aspect='auto', origin='lower', norm = "log", interpolation="none", vmin = 0.1, vmax=10)
plt.colorbar()


plt.show()