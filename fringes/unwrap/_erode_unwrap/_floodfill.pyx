# distutils: language=c++
# cython: language_level=2
import numpy as np

from fringes.scanning.pixel cimport Pixel, pixel_t, pixel_list
from fringes.scanning.floodfill cimport FloodFill, list_t
cimport cython

ctypedef unsigned char uint8_t
ctypedef (int, int) point

cdef inline double unwrap_value(double v1, double v2):
    wrap_diff = round(v2 - v1)
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
    cdef Pixel start_pixel = Pixel(start_at[0], start_at[1])
    cdef FloodFill scanner = FloodFill(pp.shape, start_pixel, mask)
    cdef uint8_t[:, :] visited = np.zeros(pp.shape, dtype=np.uint8)
    up = np.zeros_like(pp)
    cdef mview_up = up
    cdef mview_pp = pp
    cdef mview_mask = mask
    cdef list neighbors

    mview_up[start_pixel._pixel.row, start_pixel._pixel.col] = mview_pp[start_pixel._pixel.row, start_pixel._pixel.col]
    neighbors = filter_neighborhood(start_pixel._neighborhood(False), mview_mask, visited)
    cdef pixel_t neighbor
    for neighbor in neighbors:  # type: Pixel
        mview_up[neighbor.row, neighbor.col] = unwrap_value(mview_up[start_pixel._pixel.row,
                                                                                   start_pixel._pixel.col],
                                                      mview_pp[neighbor.row, neighbor.col])
    visited[start_pixel._pixel.row, start_pixel._pixel.col] = True
    cdef Pixel pixel
    while not scanner.empty():
        pixel = scanner._pop_pixel()
        neighbors = filter_neighborhood(pixel._neighborhood(False), mview_mask, visited)
        for neighbor in neighbors:  # type: Pixel
            mview_up[neighbor.row,
                     neighbor.col] = unwrap_value(mview_up[pixel.row, pixel.col],
                                                  mview_pp[neighbor.row, neighbor.col])
    return up
