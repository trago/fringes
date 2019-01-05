from typing import List

import numpy as np
from numba import jit

from ..scanning import Pixel


def filter_neighborhood(neighbors: List[Pixel], mask: np.ndarray, visited: np.ndarray) -> List[Pixel]:
    valid_neighbors: List[Pixel] = []
    mm, nn = mask.shape
    for pixel in neighbors:  # type: Pixel
        if 0 <= pixel.row < mm and 0 <= pixel.col < nn:
            if mask[pixel.row, pixel.col]:
                if not visited[pixel.row, pixel.col]:
                    visited[pixel.row, pixel.col] = True
                    valid_neighbors.append(pixel)
                    # print(pixel)
    return valid_neighbors


def absolute_gradient(p: np.ndarray, mask: np.ndarray = None) -> (np.ndarray, np.ndarray):
    if mask is None:
        mask = np.ones_like(p, np.uint8)
    return _absolute_gradient(p, mask)


# In numba I can't find a form to make a signature that returns a tuple of numeric objects like arrays.
# Need some signature like the following: '(f8[:,:], f8[:,:])(f8[:,:], u1[:,:])'
@jit(cache=True, nopython=True, parallel=True)
def _absolute_gradient(_p, _mask):
    mm, nn = _p.shape

    gx = np.zeros_like(_p)
    gy = np.zeros_like(_p)
    for n in range(1, nn):
        for m in range(0, mm):
            if _mask[m, n] and _mask[m, n - 1]:
                gx[m, n] = np.fabs(_p[m, n] - _p[m, n - 1])
    for n in range(0, nn):
        for m in range(1, mm):
            if _mask[m, n] and _mask[m - 1, n]:
                gy[m, n] = np.fabs(_p[m, n] - _p[m - 1, n])

    return gx, gy
