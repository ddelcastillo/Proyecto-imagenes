#%% Packages
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

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from sklearn.feature_extraction.image import PatchExtractor

#%% Constants
image_list = os.listdir(os.path.join('images'))
#%% Train, validation, test: 60/20/20 split.
random_state_seed = 42
X_train, X_test = train_test_split(image_list, train_size=0.60, test_size=0.40, random_state=random_state_seed)
X_test, X_val = train_test_split(X_test, train_size=0.5, test_size=0.5, random_state=random_state_seed)
#%%
X_train_positive = []
temp_image = None
for image_file in X_train:
    tree = ET.parse(os.path.join('annotations', image_file.split('.')[0] + '.xml'))
    root = tree.getroot()
    temp_image = cv.imread(os.path.join('images', image_file), cv.IMREAD_GRAYSCALE)
    for tags in root.findall('object'):
        box = tags.find('bndbox')
        xmin, xmax, ymin, ymax = int(box.find('xmin').text), int(box.find('xmax').text), int(
            box.find('ymin').text), int(
            box.find('ymax').text)
        X_train_positive.append(temp_image[ymin:ymax, xmin:xmax])

#%% Getting positive HOG-SVM features
progress_bar = tqdm(total=len(X_train_positive))
X_train_positive_hog = []
for positive in X_train_positive:
    X_train_positive_hog.append(hog(resize(positive, (128, 128)), orientations=9, visualize=False))
    progress_bar.update(1)
X_train_positive_hog = np.array([np.array(row) for row in X_train_positive_hog])
#%% Saving positive HOG-SVM features
np.save('hog_positive_features.npy', X_train_positive_hog)
#%% Extracting negative features
negative_image_types = ['brick', 'camera', 'chelsea', 'coffee', 'coins', 'horse', 'moon']
negative_images = [rgb2gray(getattr(data, image_type)()) for image_type in negative_image_types]


#%% Negative samples from unrelated image patches


def extract_patches(image, scale=1.0):
    patch_size = tuple((scale * np.array((16, 16))).astype(int))
    pe = PatchExtractor(patch_size=patch_size, max_patches=400, random_state=random_state_seed)
    patches = pe.transform(image[np.newaxis])
    patches = np.array([resize(patch, (128, 128)) for patch in patches])
    return patches


X_train_negative = np.vstack([extract_patches(negative_image) for negative_image in negative_images])
#%% Getting negative HOG-SVM features
progress_bar = tqdm(total=len(X_train_negative))
X_train_negative_hog = []
for negative in X_train_negative:
    X_train_negative_hog.append(hog(negative, orientations=9, visualize=False))
    progress_bar.update(1)
X_train_negative_hog = np.array([np.array(row) for row in X_train_negative_hog])
#%% Saving negative HOG-SVM features
np.save('hog_negative_features.npy', X_train_negative_hog)
#%% Loading positive and negative HOG-SVM features
X_train_positive_hog = np.load('hog_positive_features.npy')
X_train_negative_hog = np.load('hog_negative_features.npy')
#%% Preparing features for training
X_train_hog = np.vstack([X_train_positive_hog, X_train_negative_hog])
Y_train_hog = np.concatenate(
    (np.ones(X_train_positive_hog.shape[0]), np.negative(np.ones(X_train_negative_hog.shape[0])))).astype(np.int_)
np.save('hog_X_train.npy', X_train_hog)
np.save('hog_Y_train.npy', Y_train_hog)
#%% Train SVM /!\ WILL TAKE A LONG TIME TO EXECUTE! /!\
svc = svm.SVC()
svc_parameters = {'kernel': ['linear', 'sigmoid', 'rbf'], 'C': [1, 2, 4, 8]}
clf = GridSearchCV(svc, svc_parameters)
clf.fit(X_train_hog, Y_train_hog)
#%% Saving grid results
joblib.dump(clf, 'hog_svm_grid_results.joblib')
# Best score: 0.9622274289011479
# Best parameters: {'C': 2, 'kernel': 'rbf'}
#%% Importing data and best results
clf = joblib.load('grid_results.joblib')
results_df = pd.DataFrame(clf.cv_results_)
results_df.sort_values(by=['rank_test_score'])
results_df = (
    results_df.set_index(results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis(
        'kernel'))
print(results_df[['params', 'rank_test_score', 'mean_test_score', 'std_test_score']])
#%% HOG-SVM detector built with the best parameters
X_train_hog = np.load('hog_X_train.npy')
Y_train_hog = np.load('hog_Y_train.npy')
model_svm = clf.best_estimator_
model_svm.fit(X_train_hog, Y_train_hog)
#%% Saving model
joblib.dump(model_svm, 'hog_svm_model.joblib')
