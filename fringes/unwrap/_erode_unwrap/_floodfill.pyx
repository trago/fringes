import numpy as np

from fringes.scanning.pixel cimport Pixel, pixel_t, pixel_list
from fringes.scanning.floodfill cimport FloodFill, list_t
cimport cython

ctypedef unsigned char uint8_t

cdef double unwrap_value(double v1, double v2):
    wrap_diff = round(v2 - v1)
    return v2 - wrap_diff

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef list filter_neighborhood(list neighbors, uint8_t[:, :] mask, uint8_t[:, :] visited):
    cdef list valid_neighbors = []
    cdef int mm = mask.shape[0]
    cdef int nn = mask.shape[1]
    cdef Pixel pixel
    for pixel in neighbors:  # type: Pixel
        if 0 <= pixel._pixel.row < mm and 0 <= pixel._pixel.col < nn:
            if mask[pixel._pixel.row, pixel._pixel.col]:
                if not visited[pixel._pixel.row, pixel._pixel.col]:
                    visited[pixel._pixel.row, pixel._pixel.col] = True
                    valid_neighbors.append(pixel)
                    # print(pixel)
    return valid_neighbors

# Note: I can't declare array arguments as memoryviews because it marks an error of wrong number of dimensions
#       when calling this function
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def floodfill_unwrap(pp, mask = None, start_at = (50, 50)):
    if mask is None:
        mask = np.ones(pp.shape, dtype=np.uint8)
    cdef Pixel start_pixel = Pixel(start_at[0], start_at[1])
    cdef FloodFill scanner = FloodFill(pp.shape, start_pixel, mask)
    cdef uint8_t[:, :] visited = np.zeros(pp.shape, dtype=np.uint8)
    up = np.zeros_like(pp)
    cdef mview_up = up
    cdef mview_pp = pp
    cdef mview_mask = mask

    mview_up[start_pixel._pixel.row, start_pixel._pixel.col] = mview_pp[start_pixel._pixel.row, start_pixel._pixel.col]
    neighbors = filter_neighborhood(start_pixel.neighborhood(shuffle=True), mview_mask, visited)
    cdef Pixel neighbor
    for neighbor in neighbors:  # type: Pixel
        mview_up[neighbor._pixel.row, neighbor._pixel.col] = unwrap_value(mview_up[start_pixel._pixel.row,
                                                                                   start_pixel._pixel.col],
                                                      mview_pp[neighbor._pixel.row, neighbor._pixel.col])
    visited[start_pixel._pixel.row, start_pixel._pixel.col] = True
    cdef Pixel pixel
    while True:
        try:
            pixel = scanner._next_pixel()
            neighbors = filter_neighborhood(pixel.neighborhood(), mview_mask, visited)
            for neighbor in neighbors:  # type: Pixel
                mview_up[neighbor._pixel.row,
                         neighbor._pixel.col] = unwrap_value(mview_up[pixel._pixel.row, pixel._pixel.col],
                                                             mview_pp[neighbor._pixel.row, neighbor._pixel.col])
        except StopIteration:
            break
    return up
