import unittest
from skimage.viewer import ImageViewer, CollectionViewer
from fringes.unwrap.methods import unwrap_value
from fringes.unwrap import floodfill
from fringes.simulations import wavefront
from fringes.operators.basic import normalize_range

class erode_unwrap(unittest.TestCase):

    def test_unwrap_value(self):
        v1 = 12.4
        v2 = 1.2

        v1 = unwrap_value(v2, v1)

        print(v2)

        self.assertTrue(abs(v2 - v1) < 0.5)

    def test_unwrap(self):

        phase = wavefront((256, 512), {'parabola': 0.0005}, noise={'normal':(0, 0.5)}, normalize=True)
        up = floodfill(phase, start_at=(128, 256))

        viewer = CollectionViewer([normalize_range(up), normalize_range(phase)])
        viewer.show()
