from typing import Tuple

import numpy as np
from numba import jit
from skimage.morphology import binary_erosion

from ._utils import absolute_gradient
from .methods import unwrap_value
from ..operators.basic import wrap
from ._erode_unwrap import floodfill_unwrap


# See numba doc: http://numba.pydata.org/numba-doc/latest/reference/types.html#numba-types
#
# Notes
# -----
#   1. The parameter `mask` has default argument value as `None`. In numba we could use `optional`,
#      however, by using `optional` we can not use `nopython=True`. This is why we wrap the function
#      in this way to obtain the function signature as we want
#   2. Numba can access global variables within its context. For example here, the function _find_inconsistencies
#      can access variable pw from its container function. Doing this, numba can not generate cache even though
#      cache=True is set.
#
def find_inconsistencies(pw: np.ndarray, mask: np.ndarray = None):
    '''
    Generates a mask with inconsistencies found in `pw`.

    Inconsistencies are marked with 0's and consistent with 1's

    Parameters
    ----------
    pw: array_like
        2D array
    mask: array_like
        2D array

        The inconsistencies marked are whithin the masked region

    Returns
    -------
    array_like
        2D array mask with inconsistent pixels marked as 0's, 1's otherwise

    Examples
    --------
    >>> from skimage.viewer import CollectionViewer
    >>> from fringes.operators.basic import normalize_range
    >>> from fringes.simulations import wavefront
    >>> phase = wavefront((256, 512), {'peaks': 15}, noise={'normal': (0, 0.5)}, normalize=True)
    >>> mask = find_inconsistencies(phase)
    >>> up = floodfill_unwrap(phase, mask, start_at=(128, 256))
    >>> viewer = CollectionViewer([normalize_range(mask),
    >>>                               normalize_range(up),
    >>>                               normalize_range(phase)])
    >>> viewer.show()
    '''

    @jit(['u1[:,:](f8[:,:], u1[:,:])',
          'u1[:,:](f4[:,:], u1[:,:])'],
         cache=True, nopython=True, parallel=True)
    def _find_inconsistencies(_pw: np.ndarray, _mask: np.ndarray):
        mm, nn = _pw.shape

        inconsistent = np.ones_like(_pw, dtype=np.uint8)
        for n in range(1, nn - 1):
            for m in range(1, mm - 1):
                if _mask[m, n] and _mask[m, n - 1] \
                        and _mask[m - 1, n - 1] and _mask[m - 1, n] \
                        and _mask[m + 1, n + 1] and _mask[m + 1, n] and _mask[m, n + 1]:
                    s = wrap(_pw[m, n - 1] - _pw[m, n])
                    s += wrap(_pw[m, n] - _pw[m, n + 1])
                    s += wrap(_pw[m, n + 1] - _pw[m, n - 1])
                    if np.fabs(s) > 1e-8:
                        for j in range(n - 1, n + 2):
                            inconsistent[m, j] = 0

                    s = wrap(_pw[m - 1, n] - _pw[m, n])
                    s += wrap(_pw[m, n] - _pw[m + 1, n])
                    s += wrap(_pw[m + 1, n] - _pw[m - 1, n])
                    if np.fabs(s) > 1e-8:
                        for j in range(m - 1, m + 2):
                            inconsistent[j, n] = 0

                    s = wrap(_pw[m - 1, n - 1] - _pw[m, n])
                    s += wrap(_pw[m, n] - _pw[m + 1, n + 1])
                    s += wrap(_pw[m + 1, n + 1] - _pw[m - 1, n - 1])
                    if np.fabs(s) > 1e-8:
                        inconsistent[m - 1, n - 1] = 0
                        inconsistent[m, n] = 0
                        inconsistent[m + 1, n + 1] = 0

                    s = wrap(_pw[m + 1, n - 1] - _pw[m, n])
                    s += wrap(_pw[m, n] - _pw[m - 1, n + 1])
                    s += wrap(_pw[m - 1, n + 1] - _pw[m + 1, n - 1])
                    if np.fabs(s) > 1e-8:
                        inconsistent[m + 1, n - 1] = 0
                        inconsistent[m, n] = 0
                        inconsistent[m - 1, n + 1] = 0

                    # s = wrap(_pw[m - 1, n - 1] - _pw[m - 1, n])
                    # s += wrap(_pw[m - 1, n] - _pw[m - 1, n + 1])
                    # s += wrap(_pw[m - 1, n + 1] - _pw[m, n + 1])
                    # s += wrap(_pw[m, n + 1] - _pw[m + 1, n + 1])
                    # s += wrap(_pw[m + 1, n + 1] - _pw[m + 1, n])
                    # s += wrap(_pw[m + 1, n] - _pw[m + 1, n - 1])
                    # s += wrap(_pw[m + 1, n - 1] - _pw[m, n - 1])
                    # s += wrap(_pw[m, n - 1] - _pw[m - 1, n - 1])
                    # if np.fabs(s) > 1e-8:
                    #     inconsistent[m - 1, n - 1] = 0
                    #     inconsistent[m - 1, n] = 0
                    #     inconsistent[m - 1, n - 1] = 0
                    #     inconsistent[m, n + 1] = 0
                    #     inconsistent[m + 1, n + 1] = 0
                    #     inconsistent[m + 1, n] = 0
                    #     inconsistent[m + 1, n - 1] = 0
                    #     inconsistent[m, n - 1] = 0

                    s = wrap(_pw[m, n] - _pw[m, n - 1])
                    s += wrap(_pw[m, n - 1] - _pw[m - 1, n - 1])
                    s += wrap(_pw[m - 1, n - 1] - _pw[m - 1, n])
                    s += wrap(_pw[m - 1, n] - _pw[m, n])
                    if np.fabs(s) > 1e-8:
                        inconsistent[m, n] = 0
                        inconsistent[m, n - 1] = 0
                        inconsistent[m - 1, n - 1] = 0
                        inconsistent[m - 1, n] = 0
        return inconsistent

    if mask is None:
        mask = np.ones(pw.shape, dtype=np.uint8)
    return _find_inconsistencies(pw, mask)


# Note: binary_erosion can not manage inplace erosion. `binary_erosion(mask, mask)` marks `MemoryError`
#
#   I can't make work the following:
#
#   >>> out = out.astype(np.bool, order='C', copy=False)
#   >>> binary_erosion(mask, out).astype(np.uint8, order='C', copy=False)
#   Raises a MemoryError
def erode_mask(mask: np.ndarray, iters: int = 1):
    selem = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=np.uint8)
    out = binary_erosion(mask, selem)  # type: np.ndarray
    for niter in range(1, iters):
        out = binary_erosion(out, selem)
    return out.astype(np.uint8, order='C', copy=False)


def dilating_unwrap(pp: np.ndarray, mask: np.ndarray = None, start_at: Tuple[int, int] = (0, 0), max_iters: int = 30):
    if mask is None:
        mask = np.ones_like(pp, np.uint8)
    inconsistencies = find_inconsistencies(pp, mask)
    new_mask = inconsistencies.copy()
    new_mask[mask == 0] = 0
    up = floodfill_unwrap(pp, new_mask, start_at)
    dx, dy = absolute_gradient(up, new_mask)
    print(dx.max(), dy.max())
    for niter in range(1, max_iters):
        if dx.max() < 0.5 and dy.max() < 0.5:
            break
        inconsistencies = erode_mask(inconsistencies)
        new_mask[inconsistencies == 0] = 0
        new_mask[mask == 0] = 0
        up = floodfill_unwrap(pp, new_mask, start_at)
        dx, dy = absolute_gradient(up, new_mask)
        print(dx.max(), dy.max())

    return up, inconsistencies


@jit(cache=True, nopython=False)
def unwrap_shell(pp, up, visited, mask):
    (mm, nn) = visited.shape
    found = False
    visited_copy = visited.copy()
    for n in range(nn):
        for m in range(mm):
            if visited[m, n] == 0 and mask[m, n]:
                if m + 1 >= 0:
                    if visited[m - 1, n] and mask[m - 1, n]:
                        found = True
                        up[m, n] = unwrap_value(up[m - 1, n], pp[m, n])
                        visited_copy[m, n] = 1.0
                if n + 1 < nn:
                    if visited[m, n + 1] and mask[m, n + 1]:
                        found = True
                        up[m, n] = unwrap_value(up[m, n + 1], pp[m, n])
                        visited_copy[m, n] = 1.0
                if m - 1 < mm:
                    if visited[m + 1, n] and mask[m + 1, n]:
                        found = True
                        up[m, n] = unwrap_value(up[m + 1, n], pp[m, n])
                        visited_copy[m, n] = 1.0
                if n - 1 >= 0:
                    if visited[m, n - 1] and mask[m, n - 1]:
                        found = True
                        up[m, n] = unwrap_value(up[m, n - 1], pp[m, n])
                        visited_copy[m, n] = 1.0
    return visited_copy, found


def erode_unwrap(pp: np.ndarray, up: np.ndarray, inconsistencies, mask=None, iters=-1):
    if mask is None:
        mask = np.ones_like(inconsistencies)
    erode_inconsistencies, found = unwrap_shell(pp, up, inconsistencies, mask)
    up_new = up.copy()

    n = 0
    while found:
        erode_inconsistencies, found = unwrap_shell(pp, up_new, erode_inconsistencies, mask)
        n += 1
        if iters > 0:
            if n > iters:
                break

    return up_new, erode_inconsistencies
