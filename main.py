import src.integral_image as integral
import src.haar_features as haar
import src.adaboost as ab
import src.utils as utils
import numpy as np


if __name__ == "__main__":
    pos_image_path = './train_images/FACES'
    neg_image_path = './train_images/NFACES'

    # load images
    print("\nloading image samples from files ...")
    pos_images = utils.load_images(pos_image_path)
    neg_images = utils.load_images(neg_image_path)
    print("\nthe number of pos samples loaded: %d\nthe number of neg samples loaded: %d" % 
        (len(pos_images), len(neg_images)))

    # images partition
    print("\npartitioning images into train/dev/test sets ...")
    pos_train_imgs, pos_dev_imgs, pos_test_imgs = utils.images_partition(pos_images)
    neg_train_imgs, neg_dev_imgs, neg_test_imgs = utils.images_partition(neg_images)
    num_train, num_dev, num_test = len(pos_train_imgs), len(pos_dev_imgs), len(pos_test_imgs)
    print("\nthe number of training set: %d\nthe number of development set: %d\nthe number of test set: %d" % 
        (num_train, num_dev, num_test))

    # integral images
    pos_train_int_imgs, neg_train_int_imgs = list(), list()
    pos_train_variance, neg_train_variance = list(), list()

    print("\ngetting integral images ...")
    for i in range(num_train):
        int_img_pos, var_pos = integral.IntegralImage(pos_train_imgs[i]).get_integral_image()
        int_img_neg, var_neg = integral.IntegralImage(neg_train_imgs[i]).get_integral_image()

        pos_train_int_imgs.append(int_img_pos)
        pos_train_variance.append(var_pos)
        neg_train_int_imgs.append(int_img_neg)
        neg_train_variance.append(var_neg)
    print("\nintegral images obtained")

    # get the variance right
    pos_train_variance, neg_train_variance = np.sqrt(pos_train_variance), np.sqrt(neg_train_variance)

    # parameters
    num_classifier = 2
    min_feature_height = 8
    max_feature_height = 10
    min_feature_width = 8
    max_feature_width = 10

    print("\nAdaBoost begins ...")
    classifiers = ab.learn(pos_train_int_imgs, neg_train_int_imgs, num_classifier, 
        min_feature_width, max_feature_width, min_feature_height, max_feature_height, verbose=True)
    
    print("\nwriting classifiers into json file for later use")
    utils.write_json_file(classifiers)


