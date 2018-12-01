import unittest
from src.integral_image import IntegralImage as integral
from src.integral_image import get_sum
import numpy as np
from PIL import Image

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.integral_image_test
'''


class IntegralImageTest(unittest.TestCase):

    def setUp(self):
        # method called to prepare the test fixture
        self.ori_img = np.array(Image.open(
            './train_images/FACES/face00001.bmp'), dtype=np.float64)
        self.int_img = integral(self.ori_img).int_img

    def tearDown(self):
        # method called immediately after the test method has been called and the result recorded
        pass

    def test_integral_calculation(self):
        # top-left corner
        assert self.int_img[1, 1] == self.ori_img[0, 0] 
        # bottom-left corner
        assert self.int_img[-1, 1] == np.sum(self.ori_img[:, 0])
        # top-right corner
        assert self.int_img[1, -1] == np.sum(self.ori_img[0, :])
        # bottom-right corner
        assert self.int_img[-1, -1] == np.sum(self.ori_img)

    def test_get_sum(self):
        # pay attention that integral image has additional rows or columns of 0
        assert get_sum(self.int_img, (0, 0), (1, 1)) == self.ori_img[0, 0]
        assert get_sum(self.int_img, (0, 0), (-1, -1)) == np.sum(self.ori_img)


if __name__ == "__main__":
    # a command-line program that loads a set of tests from integral_image and runs them
    unittest.main()
