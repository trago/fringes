import unittest
from fringes.scanning import FloodFill, Pixel
import numpy as np

class FloodFill_Test(unittest.TestCase):

    def test_init(self):
        scanner = FloodFill((10, 10), Pixel(5, 5))
        self.assertFalse(scanner.empty())

    def test_iter_1(self):
        visited = np.zeros((10, 10), dtype=np.uint8)
        visited[5, 5] = 1

        scanner = FloodFill((10, 10), Pixel(5, 5))
        count =1
        for pix in scanner:
            if count <= 28:
                visited[pix.col, pix.row] = count
                count += 1
            else:
                break

        self.assertTrue(visited[4:7, 4:7].all() >= 1)

        print(visited)

    def test_iter_2(self):
        visited = np.zeros((1512, 512), dtype=np.uint32)
        visited[5, 5] = 1

        scanner = FloodFill((1512, 512), Pixel(5, 5))
        count = 1
        for pix in scanner:
            visited[pix.row, pix.col] = count
            count += 1

        self.assertTrue(count == 1512*512)
