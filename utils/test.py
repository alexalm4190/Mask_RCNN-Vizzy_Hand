import numpy as np
from scipy import ndimage

a = np.zeros((10, 10, 1), dtype = "bool")
a[3:7, 4] = 1
a[3:7, 5] = 1
a[3:7, 6] = 1
print(np.logical_not(a))
print(np.sum(a, axis=None))
a = np.squeeze(a)
print(np.argwhere(a==1))

c_y = np.gradient(a.astype(float), axis=0)
c_x = np.gradient(a.astype(float), axis=1)
print(np.sqrt( np.square(c_y) + np.square(c_x) ))

d = np.logical_xor(a, ndimage.morphology.binary_erosion(a))
print(d)