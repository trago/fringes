from unittest import TestCase

from skimage.viewer import ImageViewer, CollectionViewer

from fringes import simulations
from fringes.operators.basic import normalize_range


class interferogram(TestCase):

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


class interferogram_psi(TestCase):

    def test_01(self):
        print('Test 01')
        phase = {'ramp': (5, 6)}
        dc = {'gaussian': (130, 100.0)}
        noise = {'normal': (0, 0.1)}
        magn = {'gaussian': (130, 20)}
        steps = (5, 3.1416/2)

        imgs = simulations.interferogram_psi(steps, (512, 512), dc, phase, magn, noise)

        viewer = CollectionViewer(imgs)
        viewer.show()

    def test_02(self):
        print('Test 02')
        phase = {'peaks': 15.3}
        dc = {'gaussian': (100.0, 1000)}
        noise = 'speckle'
        magn = {'gaussian': (130, 100)}
        steps = (5, 3.1416/2)

        imgs = simulations.interferogram_psi(steps, (512, 512), dc, phase, magn, noise)

        viewer = CollectionViewer(imgs)
        viewer.show()

    def test_03(self):
        print('Test 03')
        phase = {'parabola': 0.0005}
        dc = {'gaussian': (100.0, 100)}
        noise = 'speckle'
        magn = {'gaussian': (130, 100)}
        steps = [0.1, 0.5, 0.9, 1.5, 2.2, 3.1, 3.9]

        imgs = simulations.interferogram_psi(steps, (512, 512), dc, phase, magn, noise)

        viewer = CollectionViewer(imgs)
        viewer.show()
