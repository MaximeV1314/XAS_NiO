import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from triqs.gf import *
from triqs.operators import *
from triqs.plot.mpl_interface import subplots
from h5 import HDFArchive

##################      path, subplots creation      ##################

plot_nofit = False
it = None

if it == None:
    path = "results/"
    file = "DMFT_U7.00_beta40.00.h5"
else:
    path = "results_save/"
    file = "DMFT_U7.00_beta40.00_it%d.h5"%it

fig, (t, b) = subplots(2, 4, figsize = (28, 12))  # 9 couleurs progressivement
plt.subplots_adjust(wspace=0.3, hspace=0.05)

ax_in_t0 = t[0].inset_axes([0.3, 0.45, 0.5, 0.5])
ax_in_b0 = b[0].inset_axes([0.3, 0.2, 0.5, 0.5])

ax_in_t2 = t[2].inset_axes([0.3, 0.45, 0.5, 0.5])
ax_in_b2 = b[2].inset_axes([0.3, 0.2, 0.5, 0.5])

##################          reading, plots          ##################

with HDFArchive(path + file, "r") as A :
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(A)-1))
    colors2 = plt.cm.viridis(np.linspace(0, 1, len(A)-1))

    for i in range(0, len(A)-1):
        t[0].oplot(A["iteration_%d"%i]["G_iw"]["up_1"].real, color=colors[i], label = "m = %d t2g"%i)
        ax_in_t0.oplot(A["iteration_%d"%i]["G_iw"]["up_1"].real, color=colors[i])
        b[0].oplot(A["iteration_%d"%i]["G_iw"]["up_1"].imag, color=colors[i], label = "m = %d t2g"%i)
        ax_in_b0.oplot(A["iteration_%d"%i]["G_iw"]["up_1"].imag, color=colors[i])

        t[1].oplot(A["iteration_%d"%i]["G0_iw"]["up_1"].real, color=colors[i], label = "m = %d t2g"%i)
        b[1].oplot(A["iteration_%d"%i]["G0_iw"]["up_1"].imag, color=colors[i], label = "m = %d t2g"%i)

        t[2].oplot(A["iteration_%d"%i]["Sigma_iw"]["up_1"].real, color=colors[i])
        ax_in_t2.oplot(A["iteration_%d"%i]["Sigma_iw"]["up_1"].real, color=colors[i])
        b[2].oplot(A["iteration_%d"%i]["Sigma_iw"]["up_1"].imag, color=colors[i])
        ax_in_b2.oplot(A["iteration_%d"%i]["Sigma_iw"]["up_1"].imag, color=colors[i])

        t[3].oplot(A["iteration_%d"%i]["G_tau"]["up_1"].real, color=colors[i])
        b[3].oplot(A["iteration_%d"%i]["Delta_tau"]["up_1"].real, color=colors[i])



        t[0].oplot(A["iteration_%d"%i]["G_iw"]["up_0"].real, color=colors2[i], label = "m = %d eg"%i)
        ax_in_t0.oplot(A["iteration_%d"%i]["G_iw"]["up_0"].real, color=colors2[i])
        b[0].oplot(A["iteration_%d"%i]["G_iw"]["up_0"].imag, color=colors2[i], label = "m = %d eg"%i)
        ax_in_b0.oplot(A["iteration_%d"%i]["G_iw"]["up_0"].imag, color=colors2[i])

        t[1].oplot(A["iteration_%d"%i]["G0_iw"]["up_0"].real, color=colors2[i], label = "m = %d eg"%i)
        b[1].oplot(A["iteration_%d"%i]["G0_iw"]["up_0"].imag, color=colors2[i], label = "m = %d eg"%i)

        t[2].oplot(A["iteration_%d"%i]["Sigma_iw"]["up_0"].real, color=colors2[i])
        ax_in_t2.oplot(A["iteration_%d"%i]["Sigma_iw"]["up_0"].real, color=colors2[i])
        b[2].oplot(A["iteration_%d"%i]["Sigma_iw"]["up_0"].imag, color=colors2[i])
        ax_in_b2.oplot(A["iteration_%d"%i]["Sigma_iw"]["up_0"].imag, color=colors2[i])

        t[3].oplot(A["iteration_%d"%i]["G_tau"]["up_0"].real, color=colors2[i])
        b[3].oplot(A["iteration_%d"%i]["Delta_tau"]["up_0"].real, color=colors2[i])

        if plot_nofit:
            t[2].oplot(A["iteration_%d"%i]["Sigma_iw_nofit"]["up_0"].real, color=colors2[i], alpha=0.5)
            b[2].oplot(A["iteration_%d"%i]["Sigma_iw_nofit"]["up_0"].imag, color=colors2[i], alpha=0.5)
            t[2].oplot(A["iteration_%d"%i]["Sigma_iw_nofit"]["up_1"].real, color=colors[i], alpha=0.5)
            b[2].oplot(A["iteration_%d"%i]["Sigma_iw_nofit"]["up_1"].imag, color=colors[i], alpha=0.5)

##################          xlim, y_lim          ##################

t[0].set_xlim(0, 90)
ax_in_t0.set_xlim(0, 5)
t[1].set_xlim(0, 90)
t[2].set_xlim(0, 90)
ax_in_t2.set_xlim(0, 5)
t[3].set_xlim(0, 40)

b[0].set_xlim(0, 90)
ax_in_b0.set_xlim(0, 5)
b[1].set_xlim(0, 90)
b[2].set_xlim(0, 90)
ax_in_b2.set_xlim(0, 5)
b[3].set_xlim(0, 40)

b[0].set_ylim(-0.5, 0.05)
ax_in_b0.set_ylim(-0.5, 0.05)
t[2].set_ylim(45, 65)
b[2].set_ylim(-10, 0.5)
ax_in_b2.set_ylim(-5, 0.5)

##################          set legend False          ##################

t[0].legend().set_visible(False)
ax_in_t0.legend().set_visible(False)
b[0].legend().set_visible(False)
ax_in_b0.legend().set_visible(False)
b[1].legend().set_visible(False)
b[2].legend().set_visible(False)
ax_in_b2.legend().set_visible(False)
t[2].legend().set_visible(False)
ax_in_t2.legend().set_visible(False)
b[3].legend().set_visible(False)
t[3].legend().set_visible(False)

##################          xlabel, y_label          ##################

t[0].set_xlabel("")
ax_in_t0.set_xlabel("")
t[1].set_xlabel("")
t[2].set_xlabel("")
ax_in_t2.set_xlabel("")
t[3].set_xlabel("")
b[0].set_xlabel(r"$i\omega_n$", fontsize = 15)
ax_in_b0.set_xlabel("")
b[1].set_xlabel(r"$i\omega_n$", fontsize = 15)
b[2].set_xlabel(r"$i\omega_n$", fontsize = 15)
ax_in_b2.set_xlabel("")
b[3].set_xlabel(r"$\tau$", fontsize = 15)

t[0].set_ylabel(r"Re[$G_{\uparrow}(i\omega_n)$]", fontsize = 15)
ax_in_t0.set_ylabel("")
b[0].set_ylabel(r"Im[$G_{\uparrow}(i\omega_n)$]", fontsize = 15)
ax_in_b0.set_ylabel("")
t[1].set_ylabel(r"Re[$G_{0\uparrow}(i\omega_n)$]", fontsize = 15)           # rotation=-90, labelpad=20
b[1].set_ylabel(r"Im[$G_{0\uparrow}(i\omega_n)$]", fontsize = 15)
t[2].set_ylabel(r"Re[$\Sigma_{\uparrow}(i\omega_n)$]", fontsize = 15)
ax_in_t2.set_ylabel("")
b[2].set_ylabel(r"Im[$\Sigma_{\uparrow}(i\omega_n)$]", fontsize = 15)
ax_in_b2.set_ylabel("")
t[3].set_ylabel(r"$G_{\uparrow}(\tau)$", fontsize = 15)
b[3].set_ylabel(r"$\Delta_{\uparrow}(\tau)$", fontsize = 15)

t[0].set_xticklabels([])
t[1].set_xticklabels([])
t[2].set_xticklabels([])
t[3].set_xticklabels([])

#t[1].yaxis.set_label_position("right")
#t[1].yaxis.tick_right()
#b[1].yaxis.set_label_position("right")
#b[1].yaxis.tick_right()

if it==None:
     fig.savefig("dmft_conv_Giw_G0iw.pdf", bbox_inches="tight", pad_inches=0)

else :
     fig.savefig("plot/dmft_conv_Giw_G0iw_it%d.pdf"%it, bbox_inches="tight", pad_inches=0)
