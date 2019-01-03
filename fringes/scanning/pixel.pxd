from libcpp cimport bool

cdef class Pixel:
    cdef int _col
    cdef int _row


    cpdef list neighborhood(self, bool shuffle)
