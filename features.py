import cv2
import numpy as np

# Feature extraction based on integral image
# Integral image, according to Viola & Jones, is quite simple and effective

        
def feature_ex():
    # Load image
    img = cv2.imread("./train_images/FACES/face00009.bmp", 0)
    img = cv2.resize(img, (24, 24))
    # Calculate integral image
    int_img = np.zeros(img.shape)

    # img = np.ones((24,24))
    # int_img = np.zeros(img.shape)    
    
    memo = np.zeros(img.shape) # 1 means this position already calc
    # Calculate value of pixel
    def calc(x, y):
        if x < 0 or y < 0:
            return 0
        if memo[x][y] == 1:
            return int_img[x][y]
        else:
            # principal equation
            cummulative = calc(x-1, y) + calc(x, y-1) - calc(x-1, y-1) + img[x][y]
            memo[x][y] = 1
            return cummulative

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            int_img[i][j] = calc(i, j)

    # Check integral image 
    def integral_test():
        # opencv built-in function
        int_cv_img = cv2.integral(img)
        if int_img.any() == int_cv_img[1:,1:].any():
            print("self implemented integral image work")


def test():
    feature_ex()

if __name__ == "__main__":
    feature_ex()

