from libcpp cimport bool
from libcpp.vector cimport vector
cimport cython

cdef struct pixel_t:
    int col
    int row

cdef class Pixel:
    cdef pixel_t _pixel

    cdef vector[pixel_t] neighborhood(self, bool shuffle)

