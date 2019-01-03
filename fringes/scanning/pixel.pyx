from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time, time_t
cimport cython

cdef class Pixel:

    def __init__(self, int col, int row):
        self._pixel.col = col
        self._pixel.row = row

    def __add__(self, Pixel other):
        return _add(self, other)

    def __getitem__(self, int item):
        if item == 0:
            return self._pixel.col
        if item == 1:
            return self._pixel.row
        raise IndexError('A pixels has only two elements (row, col)')

    def __str__(self):
        return '({}, {})'.format(self._pixel.col, self._pixel.col)

    def neighborhood(self, shuffle=False):
        cdef list obj_pixels = []
        cdef vector[pixel_t] pixels = self._neighborhood(shuffle)
        cdef int n
        for n in range(8):
            obj_pixels.append(Pixel(pixels[n].col, pixels[n].row))

        return obj_pixels

    @property
    def col(self):
        return self._pixel.col

    @property
    def row(self):
        return self._pixel.row

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef vector[pixel_t] _neighborhood(self, bool shuffle):
        cdef time_t t
        cdef int mm[3]
        cdef int nn[3]
        nn[:] = [-1, 0, 1]
        mm[:] = [-1, 0, 1]

        if shuffle:
            srand(<unsigned>time(&t))
            _shuffle(nn, 3)
            _shuffle(mm, 3)

        cdef vector[pixel_t] neighbors = vector[pixel_t](8)
        cdef idx = 0
        for m in range(3):
            for n in range(3):
                if mm[m] != 0 or nn[n] != 0:
                    neighbors[idx].col = self._pixel.col + mm[m]
                    neighbors[idx].row = self._pixel.row + nn[n]
                    idx+=1

        return neighbors


cdef Pixel _add(Pixel px1, Pixel px2):
    cdef int col = px1._pixel.col + px2._pixel.col
    cdef int row = px1._pixel.row + px2._pixel.row

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
