import numpy as np

# A summed-area table, as known as an integral image
# It is produced by cumulative addition of intensities on subsequent pixels in horizontal and vertical axis
# It also greatly reduce the computation of features


class IntegralImage(object):
    # Class for getting integral image
    # attribute int_img is the calculated integral image of a bigger size than original

    def __init__(self, img):
        self.shape = (img.shape[0]+1, img.shape[1]+1)
        self.img = img
        # integral image to be calculated
        self.int_img = np.ones(self.shape)
        # memo that indicates if this position already calc
        self.memo = np.zeros(self.shape)
        self.get()

    def calc(self, x, y):
        # Calculate value of each pixel
        if x == 0 or y == 0:
            return 0
        # if already calc, return value
        if self.memo[x][y] == 1:
            return self.int_img[x][y]
        else:
            # principal equation
            cummulative = self.calc(
                x-1, y) + self.calc(x, y-1) - self.calc(x-1, y-1) + self.img[x-1][y-1]
            self.memo[x][y] = 1
            return cummulative

    def get(self):
        # Get the integral image with additional rows/cols of 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.int_img[i][j] = self.calc(i, j)


def get_sum(int_img, top_left, bottom_right):
    # get summed value over a rectangle (attention to the tuple)

    top_left = (top_left[1], top_left[0])
    bottom_right = (bottom_right[1], bottom_right[0])
    # must swap the tuples since the orientation of the coordinate system
    if top_left == bottom_right:
        return int_img[top_left]
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    
    return int_img[bottom_right] + int_img[top_left] - int_img[bottom_left] - int_img[top_right]


'''
Note that the orientations of the coordiante system are opposite
-------------------> x
|   x1,y1   x2,y1
|   x1,y2   ...
|   ...
|
y

'''
