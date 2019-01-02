import numpy as np


class Pixel:
    def __init__(self, col, row):
        self._pixel = (col, row)

    def __add__(self, other):
        col = self._pixel[0] + other.col
        row = self._pixel[1] + other.row

        return Pixel(col, row)

    def __getitem__(self, item):
        return self._pixel[item]

    def neighborhood(self, shuffle=False):
        mm = np.array([-1, 0, 1], dtype=np.int32)
        nn = np.array([-1, 0, 1], dtype=np.int32)
        if shuffle:
            np.random.shuffle(mm)
            np.random.shuffle(nn)

        neighbors = []
        for m in mm:
            for n in nn:
                if m != 0 or n != 0:
                    neighbors.append(self.__add__(Pixel(m, n)))

        return neighbors

    @property
    def col(self):
        return self._pixel[0]

    @property
    def row(self):
        return self._pixel[1]

    def __str__(self):
        return '({}, {})'.format(*self._pixel)
