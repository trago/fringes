import numpy as np
from matplotlib import pylab as plt
from fringes import *


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


M, N = 512, 512
K = 3
phi = peaks(M,N)*32
step = 2*np.pi/K
delta = np.random.rand(K)*2*np.pi
print(delta)

background = gaussian(60)*3.7
contrast = 0.1 + gaussian(100)
images = [(background + contrast*np.cos(phi + dd)) for dd in delta]

matrix_images = create_matrix(images)
matrix_V, matrix_U = vu_factorization(matrix_images, error_accuracy=1e-8, max_iters=200, verbose=True)
pp = calc_phase(matrix_V).reshape((M, N))

magn = np.absolute(matrix_V[:, 1] + 1j*matrix_V[:, 2]).reshape((M,N))

plt.figure()
plt.subplot(121)
plt.imshow(images[0], cmap=plt.cm.gray)
plt.subplot(122)
plt.imshow(images[1], cmap=plt.cm.gray)

plt.figure()
plt.subplot(121)
plt.imshow(pp, cmap=plt.cm.gray)
plt.subplot(122)
plt.imshow(magn, cmap=plt.cm.gray)

plt.show()