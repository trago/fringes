import numpy as np
from matplotlib import pylab as plt
from fringes import demodulate_psi


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
delta = np.random.rand(K)*2*np.pi

background = gaussian(60)*3.7
contrast = gaussian(100)*3.7
images = [(background + contrast*np.cos(phi + dd)) for dd in delta]

pp = demodulate_psi(images, delta)

plt.figure()
plt.subplot(121)
plt.imshow(images[0], cmap=plt.cm.gray)
plt.subplot(122)
plt.imshow(images[1], cmap=plt.cm.gray)

plt.figure()
plt.imshow(pp, cmap=plt.cm.gray)


plt.show()