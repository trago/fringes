import unittest
from fringes.scanning import pixel


class Pixel(unittest.TestCase):
    def test_init(self):
        pix = pixel.Pixel(10, 12)
        self.assertTrue(pix.col == 10 and pix.row == 12)

    def test_add(self):
        pix1 = pixel.Pixel(10, 5)
        pix2 = pixel.Pixel(10, 5)
        pix = pix1 + pix2

        self.assertTrue(pix.col == 20)
        self.assertTrue(pix.row == 10)

    def test_get(self):
        pix = pixel.Pixel(10, 15)

        self.assertTrue(pix[0] == 10)
        self.assertTrue(pix[1] == 15)


if __name__ == '__main__':
    unittest.main()
