import h5py, numpy as np
with h5py.File("real_v1.h5", "r") as f:          # RadioML 2018 opens the same way
    X, Y, Z = f["X"][:], f["Y"][:], f["Z"][:]   # RadioML is ~21 GB: read a slice, e.g. f["X"][:100000]
    classes = list(f.attrs.get("classes", []))  # RadioML has no class list inside; it is in its readme
if Y.ndim == 2: Y, Z = Y.argmax(1), Z[:, 0]     # RadioML 2018: one-hot Y, Z of shape (N, 1)
iq = X[..., 0] + 1j * X[..., 1]                 # complex clips, shape (N, 1024)
print(X.shape, classes, np.unique(Z))
