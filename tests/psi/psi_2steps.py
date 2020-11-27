"""
Numerical simulation for testing the PSI-VU factorization.
"""

import numpy as np
from matplotlib import pylab as plt
from fringes.psi import *
from fringes.simulations import functions
from skimage.filters import gaussian
from fringes.psi import psi_vu2


def linear_map(image, min, max, a, b):
    return (image - min) * (b - a) / (max - min) + a


def create_matrix(dc, image_list):
    if (len(image_list) == 2):
        image_list.insert(0, dc)
    else:
        image_list[0] = dc
    image_list = map(lambda image: image.flatten(), image_list)
    matrix = np.vstack(tuple(image_list)).T

    return matrix


def calc_phase(matrix_I, delta):
    matrix_Up = np.array([(1, 1, 1), (0, np.cos(0), np.cos(delta)), (0, np.sin(0), np.sin(delta))])
    inv_Up = np.linalg.inv(matrix_Up)
    matrix_Vp = matrix_I @ inv_Up

    cc = matrix_Vp[:, 1]
    ss = matrix_Vp[:, 2]

    pp = np.arctan2(-ss, cc)
    magn = np.sqrt(ss ** 2 + cc ** 2)
    return magn * np.cos(pp), magn * np.sin(pp), pp


def approximate_dc(image, size=64):
    ones_response = gaussian(np.ones_like(image), size, mode='constant')
    dc = gaussian(image, size, mode='constant') / ones_response

    return dc  # *ones_response.max()


# Number of fringe patterns
K = 3
shape = (128, 256)
# Generating the phase shifts
# delta = np.random.rand(K) * 2 * np.pi
delta = [0.0, 1.36]
print(delta)

phase = functions.ramp(shape[0], shape[1], 6., 1)
phase = functions.peaks(shape[0], shape[1]) * 10
# phase = functions.parabola(shape[0], shape[1])*0.0008
dc = 1 #+ functions.gaussian(30, phase.shape)
contrast = 1.0 # + 15*functions.gaussian(60, phase.shape) + 1
noise = 0.0

image_list = [dc + contrast * np.cos(phase + d) + np.random.randn(*shape) * noise for d in delta]

dc_ = approximate_dc(image_list[1])
matrix_images = create_matrix(dc_, image_list)

matrix_V, matrix_U = psi_vu2.vu_factorization(matrix_images, error_accuracy=1e-16, verbose=True)
pp = np.arctan2(-matrix_V[:, 2], matrix_V[:, 1])
cc = np.cos(pp)
ss = np.sin(pp)

# Plotting result
pp = pp.reshape(phase.shape)
img0 = cc.reshape(phase.shape)
img1 = ss.reshape(phase.shape)

# dc_ = image_list[1] - img0*np.cos(d)

print(f"img0: ({img0.min(), img0.max()})")
print(f"img1: ({img1.min(), img1.max()})")
print(f"dc_: ({dc_.min(), dc_.max()})")

plt.figure()
plt.subplot(221)
plt.imshow(image_list[1], cmap=plt.cm.gray)
plt.subplot(222)
plt.imshow(image_list[2], cmap=plt.cm.gray)
plt.subplot(223)
plt.imshow(img0, cmap=plt.cm.gray)
plt.subplot(224)
plt.imshow(img1, cmap=plt.cm.gray)

plt.figure()
plt.subplot(211)
plt.imshow(pp, cmap=plt.cm.gray)
plt.subplot(212)
plt.imshow(dc_, cmap=plt.cm.gray)

plt.show()
