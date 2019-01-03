from pixel cimport Pixel
from libcpp cimport bool

cdef class FloodFill:
    cdef char[:, :] _mask
    cdef char[:, :] _visited
    cdef int _mm
    cdef int _nn
    cdef list _pixel_queue
    cdef Pixel _current_pix

    cdef void _start(self)
    cdef Pixel _next_pixel(self)
    cdef void _extend_pixels(self, list neighbors)
    cpdef bool empty(self)
    cdef bool _is_into(self, Pixel pix)

cdef class _Lattice:
    cdef char[:, :] _lattice

    cdef char _getitem(self, Pixel item)
    cdef void _setitem(self, Pixel key, char value)