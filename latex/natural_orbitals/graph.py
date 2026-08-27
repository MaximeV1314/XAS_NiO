import matplotlib.pyplot as plt
import numpy as np

dm_diag_N10 = np.flip(np.array([9.2406000e-05, 2.3804567e-04, 5.6181289e-04, 1.6482077e-03, 1.7301452e-03,
                        4.5655701e-01, 5.6316429e-01, 9.8335135e-01, 9.9388474e-01, 9.9890238e-01,
                        9.9986959e-01]))

dm_eigen_N10 = np.array([4.0320483e-11, 8.8126809e-13, 6.0111513e-11, 4.1486504e-07,
                          2.8514164e-03, 5.4296559e-01, 9.9762589e-01, 9.9999964e-01,
                          1.0000000e+00, 1.0000000e+00])

print(dm_diag_N10)

fig, ax = plt.subplots(figsize = (6,5))

ax.semilogy(np.min([dm_diag_N10, np.abs(1-dm_diag_N10)], axis=0), "-o", label="diag(C)")
ax.semilogy([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.min([dm_eigen_N10, np.abs(1-dm_eigen_N10)], axis=0), "-s", label=r"$\nu$")
ax.hlines(1e-6, 0, 10, colors="k", linestyles="dashed", label=r"$\epsilon = 10^{-6}$")
ax.semilogy(5, 4.5655701e-01, "ro", label=r"imp", markersize=10)

ax.set_xlabel("Index", fontsize = 16)
ax.set_ylabel(r"min(occupation, $1-$occupation)", fontsize = 16)
ax.legend(fontsize = 16)

plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

fig.tight_layout()
fig.savefig("img/occupation_N10.png", dpi = 150)
plt.show()