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
delta = [0.0, 1.]
print(delta)

phase = functions.ramp(shape[0], shape[1], 6., 1)
phase = functions.peaks(shape[0], shape[1])*10
#phase = functions.parabola(shape[0], shape[1])*0.0008
dc = functions.gaussian(30, shape) * 1.0
contrast = 1 #functions.gaussian(600, shape)*10 + 1
noise = .05

image_list = [dc + contrast * np.cos(phase + d) + np.random.randn(*shape) * noise for d in delta]

step, dc_, pp = demodulate_2frames(image_list, patch_size=10)

print(f"dc_ (min, max): ({dc_.min()}, {dc_.max()})")
print(f"dc (min, max): ({dc.min()}, {dc.max()})")
print(f"step: {np.arctan2(np.sin(step), np.cos(step))}")

# Plotting results

plt.figure()
plt.subplot(121)
plt.imshow(image_list[0], cmap=plt.cm.gray)
plt.subplot(122)
plt.imshow(image_list[1], cmap=plt.cm.gray)

plt.figure()
plt.subplot(121)
plt.imshow(pp, cmap=plt.cm.gray)
plt.subplot(122)
plt.imshow(dc_, cmap=plt.cm.gray)

plt.show()
