from unittest import TestCase

from skimage.viewer import ImageViewer

from fringes import simulations
from fringes.operators.basic import normalize_range


class phase_maps(TestCase):

    def test_interferogram_1(self):
        print('Test 01')
        phase = {'ramp': (5, 6)}
        dc = {'gaussian': (130, 1.0)}
        noise = {'normal': (0, 0.1)}
        magn = {'gaussian': (130,20)}

        img = simulations.interferogram((512, 512), dc, phase, magn, noise)

        viewer = ImageViewer(normalize_range(img))
        viewer.show()

    def test_interferogram_2(self):
        print('Test 02')

        phase = {'peaks': 13.0}
        dc = {'gaussian': (130, 1.0)}
        noise = {'normal': (0, 0.1)}

        img = simulations.interferogram((512, 512), dc, phase, noise=noise)

        viewer = ImageViewer(normalize_range(img))
        viewer.show()

    def test_interferogram_3(self):
        print('Test 03')

        phase = {'parabola': .0005}
        dc = {'gaussian': (130, 1.0)}
        noise = {'normal': (0, 0.1)}

        img = simulations.interferogram((512, 512), dc, phase, noise=noise)

        viewer = ImageViewer(normalize_range(img))
        viewer.show()

    def test_interferogram_4(self):
        print('Test 04')

        phase = {'parabola': .0005}
        dc = 'constant'
        noise = 'clean'

        img = simulations.interferogram((512, 512), dc, phase, noise=noise)

        viewer = ImageViewer(normalize_range(img))
        viewer.show()

    def test_interferogram_5(self):
        print('Test 05')

        phase = {'peaks': 14}
        dc = {'gaussian': (130, 3.0)}
        noise = 'speckle'

        img = simulations.interferogram((512, 512), dc, phase, noise=noise)

        viewer = ImageViewer(normalize_range(img))
        viewer.show()

    def test_interferogram_6(self):
        print('Test 06')
        phase = {'parabola': 0.0005}
        dc = 'constant'
        noise = {'normal': (0, 0.1)}
        magn = {'gaussian': (200,20)}

        img = simulations.interferogram((512, 512), dc, phase, magn, noise)

        viewer = ImageViewer(normalize_range(img))
        viewer.show()
