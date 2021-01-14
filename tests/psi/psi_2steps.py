"""
Numerical simulation for testing the PSI-VU factorization.
"""

import numpy as np
from matplotlib import pylab as plt
from fringes.psi.psi_vu2 import two_frames_vu
from fringes.simulations import functions
import skimage.io as ski_io

# Number of fringe patterns
K = 3
shape = (256, 512)
# Generating the phase shifts
# delta = np.random.rand(K) * 2 * np.pi
delta = [0.0, 1.57]
print(delta)

phase = functions.ramp(shape[0], shape[1], 6., 1)
phase = functions.peaks(shape[0], shape[1]) * 20
phase = functions.parabola(shape[0], shape[1]) * 0.0008
dc = 5 - functions.gaussian(200, phase.shape) * 10
contrast = 5.0 + 15 * functions.gaussian(140, phase.shape) + 1
noise = 15.5

# image_list = [dc + contrast * np.cos(phase + d) + np.random.randn(*shape) * noise for d in delta]
# pp, steps, dc0, dc1 = two_frames_phase(image_list, mask=None, dc_kernel=80, blur_kernel=.5, vu_iters=100)

I1 = ski_io.imread('../data/acetato_1.png', as_gray=True)
I2 = ski_io.imread('../data/acetato_2.png', as_gray=True) #+ functions.gaussian(150, I1.shape)
mask = ski_io.imread('../data/acetato_mask.png', as_gray=True)
image_list = (I1, I2)
pp, steps, dc0, dc1 = two_frames_vu(image_list, mask=mask, dc_kernel=32, blur_kernel=.3, vu_iters=100)

cc = np.cos(pp)
ss = np.sin(pp)

img0 = cc
img1 = ss

print(f"img0: ({img0.min(), img0.max()})")
print(f"img1: ({img1.min(), img1.max()})")
print(f"dc0: ({dc0.min(), dc0.max()})")
print(f"dc1: ({dc1.min(), dc1.max()})")
print(f"steps: {steps}")

plt.figure()
plt.subplot(221)
plt.imshow(image_list[0], cmap=plt.cm.gray)
plt.subplot(222)
plt.imshow(image_list[1], cmap=plt.cm.gray)
plt.subplot(223)
plt.imshow(dc0, cmap=plt.cm.gray)
plt.subplot(224)
plt.imshow(dc1, cmap=plt.cm.gray)

plt.figure()
plt.subplot(121)
plt.imshow(-pp, cmap=plt.cm.gray)
plt.subplot(122)
plt.imshow(img0, cmap=plt.cm.gray)


plt.show()
