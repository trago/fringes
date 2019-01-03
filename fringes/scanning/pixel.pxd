# distutils: language=c++
# cython: language_level=2
from libcpp cimport bool
from libcpp.vector cimport vector

cdef struct pixel_t:
    int col
    int row

cdef class Pixel:
    cdef pixel_t _pixel

    cdef vector[pixel_t] _neighborhood(self, bool shuffle)

