import scipy

import os
import numpy as np
import cv2

from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
# plt.style.use('seaborn')
from matplotlib.colors import ListedColormap

def SubRegionDivision(image, mode="slico", region_size=30):
    element_sets_V = []
    if mode == "slico":
        slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler = 20.0) 
        slic.iterate(20)     # The number of iterations, the larger the better the effect
        label_slic = slic.getLabels()        # Get superpixel label
        number_slic = slic.getNumberOfSuperpixels()  # Get the number of superpixels

        for i in range(number_slic):
            img_copp = image.copy()
            img_copp = img_copp * (label_slic == i)[:,:, np.newaxis]
            element_sets_V.append(img_copp)
    elif mode == "seeds":
        seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
        seeds.iterate(image,10)  # The input image size must be the same as the initialization shape and the number of iterations is 10
        label_seeds = seeds.getLabels()
        number_seeds = seeds.getNumberOfSuperpixels()

        for i in range(number_seeds):
            img_copp = image.copy()
            img_copp = img_copp * (label_seeds == i)[:,:, np.newaxis]
            element_sets_V.append(img_copp)
    return element_sets_V