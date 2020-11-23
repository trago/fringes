"""
Numerical simulation for testing the PSI-VU factorization.
"""

import numpy as np
from matplotlib import pylab as plt
from fringes.psi import *


def peaks(M, N):
    y, x = np.ogrid[-3.0:3.0:M * 1j, -3.0:3.0:N * 1j]

    p = 1 - x / 2.0 + x ** 5 + y ** 3
    p *= np.exp(-x ** 2 - y ** 2)

    return p


def gaussian(sigma=10.0, shape=(512, 512)):
    M, N = shape
    y, x = np.ogrid[-M / 2:M / 2:M * 1j, -N / 2:N / 2:N * 1j]

    g_values = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return g_values

# Dimensions
M, N = 512, 512
# Number of fringe patterns
K = 5
# Generating the phase that is going to be modulated
phi = peaks(M,N)*27
# Generating the phase shifts
delta = np.random.rand(K)*2*np.pi
print(delta)

# Modeling background illumination
background = gaussian(200)*3.7
# Modeling contrast
contrast = 1.1 + gaussian(100)
# Generating all fringe patterns
images = [(background + contrast*np.cos(phi + dd)) for dd in delta]

pp, dc = demodulate(images, error_accuracy=1e-8, max_iters=200, verbose=True)

# Plotting results
plt.figure()
plt.subplot(121)
plt.imshow(images[0], cmap=plt.cm.gray)
plt.subplot(122)
plt.imshow(images[1], cmap=plt.cm.gray)

plt.figure()
plt.subplot(121)
plt.imshow(pp, cmap=plt.cm.gray)
plt.subplot(122)
plt.imshow(dc, cmap=plt.cm.gray)

plt.show()