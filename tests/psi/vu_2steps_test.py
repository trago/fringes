"""
Numerical simulation for testing the PSI-VU factorization.
"""

import numpy as np
from matplotlib import pylab as plt
from fringes.psi import *
from fringes.simulations import functions
from skimage.transform import resize

# Number of fringe patterns
K = 3
shape = (256, 512)
# Generating the phase shifts
# delta = np.random.rand(K) * 2 * np.pi
delta = [0.0, 1.36, 2.36850563, -0.5]
print(delta)

phase = functions.ramp(shape[0], shape[1], 6., 1)
#phase += functions.peaks(shape[0], shape[1])*17
#phase += functions.parabola(shape[0], shape[1])*0.0005
dc = 1 + functions.gaussian(300, shape) * 5.0
contrast = 1.0 + functions.gaussian(300, shape)
noise = .0

image_list = [dc + contrast * np.cos(phase + d) + np.random.randn(*shape) * noise for d in delta]

step, dc_, _ = demodulate_2steps(image_list)
dc_ = resize(dc_, image_list[0].shape)

img0 = image_list[0] - dc_
img1 = image_list[1] - dc_

pp = calc_phase_2steps(img0, img1, step)

print(f"dc_ (min, max): ({dc_.min()}, {dc_.max()})")
print(f"dc (min, max): ({dc.min()}, {dc.max()})")
print(f"step: {np.arctan2(np.sin(step), np.cos(step))}")

# Plotting results

plt.figure()
plt.subplot(121)
plt.imshow(pp, cmap=plt.cm.gray)
plt.subplot(122)
plt.imshow(dc_, cmap=plt.cm.gray)

plt.show()
