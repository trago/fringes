# distutils: language=c++
# cython: language_level=2
from pixel cimport Pixel, pixel_t
from libcpp cimport bool
from libcpp.list cimport list as list_t
from libcpp.vector cimport vector

cdef class FloodFill:
    cdef char[:, :] _mask
    cdef char[:, :] _visited
    cdef int _mm
    cdef int _nn
    cdef list_t[pixel_t] _pixel_queue
    cdef pixel_t _current_pix

    cdef void _start(self)
    cdef Pixel _next_pixel(self)
    cdef void _extend_pixels(self, vector[pixel_t] neighbors)
    cpdef bool empty(self)
    cdef bool _is_into(self, pixel_t pix)
    cdef list_t[pixel_t] get_pixel_queue(self)

cdef class _Lattice:
    cdef char[:, :] _lattice

    cdef char _getitem(self, pixel_t item)
    cdef void _setitem(self, pixel_t key, char value)