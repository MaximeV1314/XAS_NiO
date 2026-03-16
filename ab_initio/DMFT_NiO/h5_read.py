from h5 import HDFArchive
import numpy as np

with HDFArchive("w90_to_triqs.h5", "r") as A:
    hopping = A["dft_input"]["hopping"]
    print(A["dft_input"])

print(hopping.shape)
print(np.diag(np.round(np.sum(hopping[:,0,:,:], axis=0)/512, 4)))
