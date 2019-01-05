import numpy as np
from numba import jit
from numba import types as tp

from ._utils import filter_neighborhood
from .methods import unwrap_value
from ..scanning import FloodFill, Pixel
from ..operators.basic import wrap


@jit(['f8[:,:](f8[:,:], optional(u1[:, :]), pyobject)',
      'f4[:,:](f4[:,:], optional(u1[:, :]), pyobject)'], forceobj=True)
def floodfill(pp: np.ndarray, mask: np.ndarray = None, start_at=(50, 50)):
    if mask is None:
        mask = np.ones(pp.shape, dtype=np.uint8)
    start_pixel = Pixel(start_at[0], start_at[1])
    scanner = FloodFill(pp.shape, start_pixel, mask)
    visited = np.zeros(pp.shape, dtype=np.uint8)
    up = np.zeros_like(pp)

    up[start_at[0], start_at[1]] = pp[start_pixel[0], start_pixel[1]]
    neighbors = filter_neighborhood(start_pixel.neighborhood(shuffle=True), mask, visited)
    for neighbor in neighbors:  # type: Pixel
        up[neighbor.row, neighbor.col] = unwrap_value(up[start_pixel.row, start_pixel.col],
                                                      pp[neighbor.row, neighbor.col])
    visited[start_pixel[0], start_pixel[1]] = True
    for pixel in scanner:  # type: Pixel
        neighbors = filter_neighborhood(pixel.neighborhood(), mask, visited)
        for neighbor in neighbors:  # type: Pixel
            up[neighbor.row, neighbor.col] = unwrap_value(up[pixel.row, pixel.col], pp[neighbor.row, neighbor.col])
    return up


# See numba doc: http://numba.pydata.org/numba-doc/latest/reference/types.html#numba-types
#
# Notes
# -----
#   1. The parameter `mask` has default argument value as `None`. In numba we could use `optional`,
#      however, by using `optional` we can not use `nopython=True`. This is why we wrap the function
#      in this way to obtain the function signature as we want.
#
def find_inconsistencies(pw: np.ndarray, mask: np.ndarray = None):
    @jit(['u1[:,:](f8[:,:], u1[:,:])',
          'u1[:,:](f4[:,:], u1[:,:])'],
         cache=True, nopython=True, fastmath=True, parallel=True)
    def _find_inconsistencies(_pw: np.ndarray, _mask: np.ndarray):
        mm, nn = _pw.shape

        inconsistent = np.ones_like(_pw, dtype=np.uint8)
        for n in range(0, nn):
            for m in range(0, mm):
                if n > 0 and m > 0:
                    if _mask[m, n] and _mask[m, n - 1] \
                            and _mask[m - 1, n - 1] and _mask[m - 1, n]:
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
