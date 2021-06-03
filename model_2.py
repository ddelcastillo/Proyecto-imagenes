# %% Packages
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
from skimage.transform.pyramids import pyramid_gaussian as pyramid

# %% Constants
image_list = os.listdir(os.path.join('images'))
global_patch_size = (128, 128)
# %% Train, validation, test: 60/20/20 split.
random_state_seed = 42
X_train, X_test = train_test_split(image_list, train_size=0.60, test_size=0.40, random_state=random_state_seed)
X_test, X_val = train_test_split(X_test, train_size=0.5, test_size=0.5, random_state=random_state_seed)
# %%
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

# %% Getting positive HOG-SVM features
progress_bar = tqdm(total=len(X_train_positive))
X_train_positive_hog = []
for positive in X_train_positive:
    X_train_positive_hog.append(
        hog(resize(positive, (global_patch_size[0], global_patch_size[1])), orientations=9, visualize=False))
    progress_bar.update(1)
X_train_positive_hog = np.array([np.array(row) for row in X_train_positive_hog])
# %% Saving positive HOG-SVM features
np.save('hog_positive_features.npy', X_train_positive_hog)
# %% Extracting negative features
negative_image_types = ['brick', 'camera', 'chelsea', 'coffee', 'coins', 'horse', 'moon']
negative_images = [rgb2gray(getattr(data, image_type)()) for image_type in negative_image_types]


# %% Negative samples from unrelated image patches


def extract_patches(image, patch_size, scale=1.0):
    sub_patch_size = tuple((scale * np.array((16, 16))).astype(int))
    pe = PatchExtractor(patch_size=sub_patch_size, max_patches=400, random_state=random_state_seed)
    patches = pe.transform(image[np.newaxis])
    patches = np.array([resize(patch, (patch_size[0], patch_size[1])) for patch in patches])
    return patches


X_train_negative = np.vstack([extract_patches(negative_image, global_patch_size) for negative_image in negative_images])
# %% Getting negative HOG-SVM features
progress_bar = tqdm(total=len(X_train_negative))
X_train_negative_hog = []
for negative in X_train_negative:
    X_train_negative_hog.append(hog(negative, orientations=9, visualize=False))
    progress_bar.update(1)
X_train_negative_hog = np.array([np.array(row) for row in X_train_negative_hog])
# %% Saving negative HOG-SVM features
np.save('hog_negative_features.npy', X_train_negative_hog)
# %% Loading positive and negative HOG-SVM features
X_train_positive_hog = np.load('hog_positive_features.npy')
X_train_negative_hog = np.load('hog_negative_features.npy')
# %% Preparing features for training
X_train_hog = np.vstack([X_train_positive_hog, X_train_negative_hog])
Y_train_hog = np.concatenate(
    (np.ones(X_train_positive_hog.shape[0]), np.negative(np.ones(X_train_negative_hog.shape[0])))).astype(np.int_)
np.save('hog_X_train.npy', X_train_hog)
np.save('hog_Y_train.npy', Y_train_hog)
# %% Train SVM /!\ WILL TAKE A LONG TIME TO EXECUTE! /!\
svc = svm.SVC()
svc_parameters = {'kernel': ['linear', 'sigmoid', 'rbf'], 'C': [1, 2, 4, 8]}
clf = GridSearchCV(svc, svc_parameters)
clf.fit(X_train_hog, Y_train_hog)
# %% Saving grid results
joblib.dump(clf, 'hog_svm_grid_results.joblib')
# Best score: 0.9622274289011479
# Best parameters: {'C': 2, 'kernel': 'rbf'}
# %% Importing data and best results
clf = joblib.load('hog_svm_grid_results.joblib')
results_df = pd.DataFrame(clf.cv_results_)
results_df.sort_values(by=['rank_test_score'])
results_df = (
    results_df.set_index(results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis(
        'kernel'))
print(results_df[['params', 'rank_test_score', 'mean_test_score', 'std_test_score']])
# %% HOG-SVM detector built with the best parameters
X_train_hog = np.load('hog_X_train.npy')
Y_train_hog = np.load('hog_Y_train.npy')
hog_svm_model = clf.best_estimator_
hog_svm_model.fit(X_train_hog, Y_train_hog)
# %% Saving model
joblib.dump(hog_svm_model, 'hog_svm_model.joblib')
# %% Loading model
hog_svm_model = joblib.load('hog_svm_model.joblib')
# %% Testing extracting patches
test_image = cv.imread(os.path.join('images', X_val[0]), cv.IMREAD_GRAYSCALE)
window_step = (32, 32)


def sliding_window(img, patch_size, i_step, j_step):
    n_i, n_j = (s for s in patch_size)
    for i in range(0, img.shape[0] - n_i, i_step):
        for j in range(0, img.shape[1] - n_i, j_step):
            patch = img[i:i + n_i, j:j + n_j]
            yield (i, j), patch


# %%
num_patches = math.ceil((test_image.shape[0] - global_patch_size[0]) / window_step[0]) * math.ceil(
    (test_image.shape[1] - global_patch_size[1]) / window_step[1])
print(f'Extracting {num_patches} patches...')
indices, patches = zip(
    *sliding_window(test_image,
                    (global_patch_size[0], global_patch_size[1]), window_step[0], window_step[1]))
print('... Done.')
# %%
try:
    progress_bar.reset()
except NameError:
    pass
progress_bar = tqdm(total=len(patches))
patches_hog = []
for i in range(len(patches)):
    patches_hog.append(hog(patches[i], orientations=9, visualize=False))
    progress_bar.update(1)
patches_hog = np.asarray(patches_hog)
print(patches_hog.shape)
# %%
y_hat = hog_svm_model.predict(patches_hog)
print(y_hat.sum())
# %%
fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')
indices = np.array(indices)
for i, j in indices[y_hat == 1]:
    ax.add_patch(plt.Rectangle((j, i), global_patch_size[0], global_patch_size[1], edgecolor='green', alpha=0.3, lw=2,
                               facecolor='none'))
for i, j in indices[y_hat == -1]:
    ax.add_patch(plt.Rectangle((j, i), global_patch_size[0], global_patch_size[1], edgecolor='red', alpha=0.3, lw=2,
                               facecolor='none'))
fig.show()
# %%
import pickle
class_model = pickle.load(open(os.path.join('modelos', 'best_model_RF.pkl'), 'rb'))
# %% Joblib
joblib.dump(class_model, 'classifier_model2.joblib')
