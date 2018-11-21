import unittest
from src.integral_image import IntegralImage as integral
import numpy as np
from PIL import Image
import cv2

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.

note that opencv library is only used to test integral image calculation
'''

class IntegralImageTest(unittest.TestCase):
    # method called to prepare the test fixture
    def setUp(self):
        self.ori_img = np.array(Image.open('./train_images/FACES/face00001.bmp'), dtype=np.float64)
        self.int_img = integral(self.ori_img).int_img

    # method called immediately after the test method has been called and the result recorded
    def tearDown(self):
        pass

    def test_integral_calculation(self):
        assert self.int_img[0,0] == self.ori_img[0,0] # top-left corner
        assert self.int_img[-1,0] == np.sum(self.ori_img[:,0]) # bottom-left corner
        assert self.int_img[0,-1] == np.sum(self.ori_img[0,:]) # top-right corner
        assert self.int_img[-1,-1] == np.sum(self.ori_img) # bottom-right corner

    def test_with_opencv(self):
        int_img_cv2 = cv2.integral(self.ori_img) # opencv built-in function
        assert self.int_img.any() == int_img_cv2[1:,1:].any()

# a command-line program that loads a set of tests from integral_image and runs them
if __name__ == "__main__":
    unittest.main()