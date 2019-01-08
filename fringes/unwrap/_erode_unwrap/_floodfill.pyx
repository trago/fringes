# distutils: language=c++
# cython: language_level=2
# distutils: sources = ./fringes/unwrap/_erode_unwrap/floodfill_unwrap.cpp
import numpy as np

from fringes.scanning.pixel cimport Pixel, pixel_t, pixel_list
from fringes.scanning.floodfill cimport FloodFill, list_t
cimport cython

ctypedef (int, int) point

cdef inline double unwrap_value(double v1, double v2):
    cdef double wrap_diff = round(v2 - v1)
    return v2 - wrap_diff

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline list filter_neighborhood(pixel_list neighbors, uint8_t[:, :] mask, uint8_t[:, :] visited):
    cdef list valid_neighbors = []
    cdef int mm = mask.shape[0]
    cdef int nn = mask.shape[1]
    cdef pixel_t pixel
    cdef int n

    for n in range(8):  # type: Pixel
        pixel = neighbors[n]
        if 0 <= pixel.row < mm and 0 <= pixel.col < nn:
            if mask[pixel.row, pixel.col]:
                if not visited[pixel.row, pixel.col]:
                    visited[pixel.row, pixel.col] = True
                    valid_neighbors.append(pixel)
                    # print(pixel)
    return valid_neighbors

# Note: I can't declare array arguments as memoryviews because it marks an error of wrong number of dimensions
#       when calling this function
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def floodfill_unwrap(pp, mask = None, point start_at = (50, 50)):
    if mask is None:
        mask = np.ones(pp.shape, dtype=np.uint8)
    up = np.zeros_like(pp)
    if not pp.flags['C_CONTIGUOUS']:
        pp = np.ascontiguousarray(pp)
    if not mask.flags['C_CONTIGUOUS']:
        mask = np.ascontiguousarray(mask)

    cdef double[:, ::1] mview_up = up
    cdef double[:, ::1] mview_pp = pp
    cdef uint8_t[:, ::1] mview_mask = mask

    _floodfill_unwrap(&mview_pp[0, 0], &mview_mask[0, 0], &mview_up[0, 0], start_at[0], start_at[1],
                      mview_pp.shape[0], mview_pp.shape[1])

    return up
