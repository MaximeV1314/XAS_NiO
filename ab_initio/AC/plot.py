import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from triqs.gf import *
from triqs.operators import *
from triqs_dft_tools.sumk_dft import *
from triqs.plot.mpl_interface import oplot

from h5 import *
import os

fig, ax = plt.subplots(1, 3, figsize=(20, 5))
dc = 52

with HDFArchive("MaxEnt_data/maxent_U7.00_beta40.00_Sigma_w_ISC.h5", "r") as A :
    Sigma_w = A["Sigma_w"]
    Sigma_w = Sigma_w - dc

with HDFArchive("MaxEnt_data/maxent_U7.00_beta40.00_G_w.h5", "r") as A :
    G_w = A["G_w"]

V2_w = (1/np.pi*(inverse(G_w) + Sigma_w)).imag

for i in range(5):
    if i==0 or i==3:
        color = 'dodgerblue'
    else :
        color = 'orangered'

    ax[0].oplot(G_w["up_%d"%i].real, "-", color = color)
    ax[0].oplot(G_w["up_%d"%i].imag, ":", color = color)

    ax[1].oplot(Sigma_w["up_%d"%i].real, "-", color = color)
    ax[1].oplot(Sigma_w["up_%d"%i].imag, ":", color = color)

    ax[2].oplot(V2_w["up_%d"%i].real, "-", color = color)

ax[0].get_legend().remove()
ax[1].get_legend().remove()
ax[2].get_legend().remove()

ax[0].legend([Line2D([0],[0],c='k',ls='-'),
           Line2D([0],[0],c='k',ls=':'),
           Line2D([0],[0],c='dodgerblue',ls='-'),
           Line2D([0],[0],c='orangered',ls='-')],
          ['Re','Im', r"$E_g$", r"$T_{2g}$"],loc="upper left")

ax[1].legend([Line2D([0],[0],c='k',ls='-'),
           Line2D([0],[0],c='k',ls=':'),
           Line2D([0],[0],c='dodgerblue',ls='-'),
           Line2D([0],[0],c='orangered',ls='-')],
          ['Re','Im', r"$E_g$", r"$T_{2g}$"],loc="upper left")

ax[2].legend([Line2D([0],[0],c='dodgerblue',ls='-'),
           Line2D([0],[0],c='orangered',ls='-')],
          [r"$E_g$", r"$T_{2g}$"],loc="upper left")

ax[0].set_ylabel(r"$G(\omega)$")
ax[1].set_ylabel(r"$\Sigma(\omega) - \mu_{dc}$")
ax[2].set_ylabel(r"$-1/\pi $ Im$(\Delta)$")

ax[2].set_xlim(-12, 7)

fig.tight_layout()
fig.savefig("spec_U7.00_beta40.00.png", dpi = 150)

plt.show()