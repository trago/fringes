# distutils: language = c++
import numpy as np
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time, time_t
cimport cython


cdef class Pixel:

    def __init__(self, int col, int row):
        self._col = col
        self._row = row

    def __add__(self, Pixel other):
        return _add(self, other)

    def __getitem__(self, int item):
        if item == 0:
            return self._col
        if item == 1:
            return self._row
        raise IndexError('A pixel has only two elements (row, col)')

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef list neighborhood(self, bool shuffle):
        cdef time_t t
        cdef int mm[3]
        cdef int nn[3]
        nn[:] = [-1, 0, 1]
        mm[:] = [-1, 0, 1]

        if shuffle:
            srand(<unsigned>time(&t))
            _shuffle(nn, 3)
            _shuffle(mm, 3)

        cdef list neighbors = []
        for m in range(3):
            for n in range(3):
                if mm[m] != 0 or nn[n] != 0:
                    neighbors.append(self.__add__(Pixel(mm[m], nn[n])))

        return neighbors

    @property
    def col(self):
        return self._col

    @property
    def row(self):
        return self._row

    def __str__(self):
        return '({}, {})'.format(*self._pixel)



cdef Pixel _add(Pixel px1, Pixel px2):
    cdef int col = px1._col + px2._col
    cdef int row = px1._row + px2._row

    return Pixel(col, row)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void _shuffle(int[:] array, size_t n) nogil:
    cdef size_t i
    cdef size_t j
    cdef int t
    if n > 1:
        for i in range(n):
            j = i + <int>(rand() / (RAND_MAX / (n - i) + 1))
            t = array[j]
            array[j] = array[i]
            array[i] = t
