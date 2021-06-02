# %%
import xml.etree.ElementTree as ET

import skimage.color
import pandas as pd
from skimage import data
import numpy
from skimage.feature import hog
from skimage.transform import resize
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import cv2 as cv
import pprint
import random
import joblib
import math
import os

from skimage.exposure import equalize_hist
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from sklearn.feature_extraction.image import PatchExtractor
from skimage.transform.pyramids import pyramid_gaussian as pyramid

# %%
with_mask_list = os.listdir(os.path.join('face_mask_dataset', 'Train', 'WithMask'))
without_mask_list = os.listdir(os.path.join('face_mask_dataset', 'Train', 'WithoutMask'))
global_patch_size = (128, 128)

# %%
test_image = cv.imread(os.path.join('face_mask_dataset', 'Train', 'WithMask', with_mask_list[0]), cv.IMREAD_GRAYSCALE)
test_image_equalized = equalize_hist(test_image)
test_image_equalized_resized = resize(test_image_equalized, global_patch_size).flatten()

# %%
with_mask = np.zeros([len(with_mask_list), global_patch_size[0]*global_patch_size[1]])
for index, image in enumerate(with_mask_list):
    test_image = cv.imread(os.path.join('face_mask_dataset', 'Train', 'WithMask', image),
                           cv.IMREAD_GRAYSCALE)
    test_image_equalized = equalize_hist(test_image)
    with_mask[index, :] = resize(test_image_equalized, global_patch_size).flatten()
# %%
without_mask = np.zeros([len(without_mask_list), global_patch_size[0]*global_patch_size[1]])
for index, image in enumerate(without_mask_list):
    test_image = cv.imread(os.path.join('face_mask_dataset', 'Train', 'WithoutMask', image),
                           cv.IMREAD_GRAYSCALE)
    test_image_equalized = equalize_hist(test_image)
    without_mask[index, :] = resize(test_image_equalized, global_patch_size).flatten()
# %%
classifier_Y_train = np.concatenate((np.ones(len(with_mask), dtype='int'), np.zeros(len(without_mask), dtype='int')))
classifier_X_train = np.vstack((with_mask, without_mask))
# %% Save
np.save('classifier_X_train.npy', classifier_X_train)
np.save('classifier_Y_train.npy', classifier_Y_train)
# %% Load
classifier_X_train = np.load('classifier_X_train.npy')
classifier_Y_train = np.load('classifier_Y_train.npy')
