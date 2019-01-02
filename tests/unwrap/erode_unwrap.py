import unittest

from fringes.unwrap.erode_unwrap import unwrap_value


class erode_unwrap(unittest.TestCase):

    def test_unwrap_value(self):
        v1 = 12.4
        v2 = 1.2

        v2 = unwrap_value(v1, v2)

        print(v2)

        self.assertTrue(abs(v2 - v1) < 0.5)

    def test_unwrap(self):
        pass
