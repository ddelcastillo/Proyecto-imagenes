# %%
from skimage.transform import resize
import matplotlib.pyplot as plt
import skimage.io as io
import pandas as pd
import numpy as np
import cv2 as cv
import os

import joblib
from skimage.exposure import equalize_hist
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from sklearn.feature_extraction.image import PatchExtractor
from skimage.transform.pyramids import pyramid_gaussian as pyramid
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
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
# %% Grid search NN
parameters = {'solver': ['lbfgs', 'adam'], 'max_iter': [300], 'hidden_layer_sizes': [1, 2, 3, 4, 5],
              'random_state': [42]}
nn = MLPClassifier()
clf_grid = GridSearchCV(nn, parameters)
clf_grid.fit(classifier_X_train, classifier_Y_train)
# %% Saving grid results
joblib.dump(clf_grid, 'classifier_grid_results.joblib')
# %%
results_df = pd.DataFrame(clf_grid.cv_results_)
results_df.sort_values(by=['rank_test_score'])
results_df = (
    results_df.set_index(results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis(
        'kernel'))
print(results_df[['params', 'rank_test_score', 'mean_test_score', 'std_test_score']])
# %%
clf = MLPClassifier(random_state=42, max_iter=300).fit(classifier_X_train, classifier_Y_train)
# %%
joblib.dump(clf, 'classifier_model.joblib')
# %% Validation sets
global_patch_size = (128, 128)
with_mask_list = os.listdir(os.path.join('face_mask_dataset', 'Validation', 'WithMask'))
with_mask = np.zeros([len(with_mask_list), global_patch_size[0]*global_patch_size[1]])
for index, image in enumerate(with_mask_list):
    test_image = cv.imread(os.path.join('face_mask_dataset', 'Validation', 'WithMask', image),
                           cv.IMREAD_GRAYSCALE)
    test_image_equalized = equalize_hist(test_image)
    with_mask[index, :] = resize(test_image_equalized, global_patch_size).flatten()
without_mask_list = os.listdir(os.path.join('face_mask_dataset', 'Validation', 'WithoutMask'))
without_mask = np.zeros([len(without_mask_list), global_patch_size[0]*global_patch_size[1]])
for index, image in enumerate(without_mask_list):
    test_image = cv.imread(os.path.join('face_mask_dataset', 'Validation', 'WithoutMask', image),
                           cv.IMREAD_GRAYSCALE)
    test_image_equalized = equalize_hist(test_image)
    without_mask[index, :] = resize(test_image_equalized, global_patch_size).flatten()
# %%
classifier_Y_validate = np.concatenate((np.ones(len(with_mask), dtype='int'), np.zeros(len(without_mask), dtype='int')))
classifier_X_validate = np.vstack((with_mask, without_mask))
# %% Saving validation
np.save('classifier_X_validate.npy', classifier_X_validate)
np.save('classifier_Y_validate.npy', classifier_Y_validate)
# %% Validation
y_hat = clf.predict(classifier_X_validate)
# %%
report = classification_report(classifier_Y_validate, y_hat, target_names=['Without mask', 'With mask'])
print(report)
# %%
