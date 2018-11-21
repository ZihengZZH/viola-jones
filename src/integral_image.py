import numpy as np

# A summed-area table, as known as an integral image
# It is produced by cumulative addition of intensities on subsequent pixels in horizontal and vertical axis
# It also greatly reduce the computation of features

class IntegralImage(object):
    def __init__(self, img):
        self.shape = img.shape
        self.img = img
        # integral image to be calculated
        self.int_img = np.ones(self.shape)
        # memo that indicates if this position already calc
        self.memo = np.zeros(self.shape)
        self.get()

    # Calculate value of each pixel
    def calc(self, x, y):
        if x < 0 or y < 0:
            return 0
        # if already calc, return value
        if self.memo[x][y] == 1:
            return self.int_img[x][y]
        else:
            # principal equation
            cummulative = self.calc(x-1, y) + self.calc(x, y-1) - self.calc(x-1, y-1) + self.img[x][y]
            self.memo[x][y] = 1
            return cummulative

    def get(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.int_img[i][j] = self.calc(i, j)
