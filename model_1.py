# %% Packages
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import cv2 as cv
import pprint
import random
import joblib
import math
import os

from tensorflow import keras
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

# %% Constants
image_list = os.listdir(os.path.join('images'))
haarcascade_frontalface_default_file = 'haarcascade_frontalface_default.xml'

# %% Face detection test
image_file_test = random.sample(image_list, 1)[0]
image_test = cv.imread(os.path.join('images', image_file_test))
image_test_gray = cv.cvtColor(image_test, cv.IMREAD_GRAYSCALE)
face_cascade = cv.CascadeClassifier(haarcascade_frontalface_default_file)
detected_faces = face_cascade.detectMultiScale(image_test_gray)
for (column, row, width, height) in detected_faces:
    cv.rectangle(
        image_test,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )
cv.imshow('Test face detection', image_test)
cv.waitKey(0)
cv.destroyAllWindows()

# %% Data preparation (should only be run once)
# Basically a Pandas df but with more dictionaries for easier access.
image_list = os.listdir(os.path.join('images'))
data = dict()
meta_data = dict()
data['Name'], data['Size'], data['Image'], data['Labels'] = [], [], [], []
meta_data['num_images'], meta_data['labels'] = 0, set()
for image_file in image_list:
    tree = ET.parse(os.path.join('annotations', image_file.split('.')[0] + '.xml'))
    root = tree.getroot()
    data['Name'].append(image_file.split('.')[0])
    data['Size'].append([root.find('size').find('width').text, root.find('size').find('height').text])
    data['Image'].append(cv.cvtColor(cv.imread(os.path.join('images', image_file)), cv.IMREAD_GRAYSCALE))
    meta_data['num_images'] += 1
    labels = []
    for tags in root.findall('object'):
        label_dict = dict()
        label_dict['name'] = tags.find('name').text
        meta_data['labels'].add(tags.find('name').text)
        box = tags.find('bndbox')
        label_dict['xmin'] = box.find('xmin').text
        label_dict['xmax'] = box.find('xmax').text
        label_dict['ymin'] = box.find('ymin').text
        label_dict['ymax'] = box.find('ymax').text
        labels.append(label_dict)
    data['Labels'].append(labels)
joblib.dump(data, 'data')
joblib.dump(meta_data, 'meta_data')

# %% Loading data
data = joblib.load('data')
meta_data = joblib.load('meta_data')

# %% Creating the model
# It's first necessary to train a model and then apply it to the data obtained.
# Dataset taken from https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory
# Loads the training data and resizes images to 128x128. Images will be in a range from
# 0 to 255, therefore a rescale factor of 1/255 is used to have values between 0 and 1.
train_image_data_generator = ImageDataGenerator(rescale=1 / 255, horizontal_flip=True)
train_data_generator = train_image_data_generator.flow_from_directory(
    directory=os.path.join('Train'), target_size=(128, 128), batch_size=2,
)
# Loads the validation data and resizes images to 128x128.
validate_image_data_generator = ImageDataGenerator(rescale=1 / 255, horizontal_flip=True)
validate_data_generator = validate_image_data_generator.flow_from_directory(
    directory=os.path.join('Validation'), target_size=(128, 128), batch_size=2,
)
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG19
# Building the network. Doesn't include the top 3 layers (recommended) and will receive
# images of the same size as those declared on the image generators: 128x128.
vgg19 = VGG19(include_top=False, input_shape=(128, 128, 3))
# Freezing the layers: state won't update during training (https://www.tensorflow.org/guide/keras/transfer_learning).
for i in vgg19.layers:
    i.trainable = False
# Creates the model, adds the vgg19 network into a sequential (linear stack of layers)
# , is indicated to flatten input, and adds a dense layer which takes 2D image and
# activates with a sigmoid function. The result is a base model.
model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))

# %% Training the model
# Based on their own guide https://www.tensorflow.org/guide/keras/train_and_evaluate
# Values from direct keras calls are bugged, using string values instead.
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'], )
# Warning: very slow operations.
history = model.fit_generator(generator=train_data_generator, validation_data=validate_data_generator, epochs=2)
# model.evaluate_generator(validate_data_generator)

# %%
model.save('model')

# %%
# model = keras.models.load_model('model')
labels = {0: 'Con máscara', 1: 'Sin máscara'}
face_cascade = cv.CascadeClassifier(haarcascade_frontalface_default_file)
p_1, p_2, r_1, r_2 = [], [], [], []  # Precision, recall.
for i in range(meta_data['num_images']):
    detected_faces = face_cascade.detectMultiScale(data['Image'][i])
    color_image = cv.cvtColor(data['Image'][i], cv.COLOR_RGB2BGR)
    n, m = int(data['Size'][i][0]), int(data['Size'][i][1])
    gt_matrix_0, gt_matrix_1 = np.zeros([n, m], dtype=np.short), np.zeros([n, m], dtype=np.short)
    a_matrix_0, a_matrix_1 = np.zeros([n, m], dtype=np.short), np.zeros([n, m], dtype=np.short)
    for (column, row, width, height) in detected_faces:
        cropped_face = np.reshape(cv.resize(color_image[row: row + height, column: column + width], (128, 128)),
                                  [1, 128, 128, 3]) / 255
        result = model.predict(cropped_face)
        result_class = labels[result.argmax()]
        if result_class == 'with_mask':
            a_matrix_0[row:(row + height), column:(column + width)] = 1
        else:
            a_matrix_1[row:(row + height), column:(column + width)] = 1
    for label in data['Labels'][i]:
        x, y, mx, my = int(label['xmin']), int(label['ymin']), int(label['xmax']), int(label['ymax'])
        if label['name'] == 'with_mask':
            gt_matrix_0[x:mx, y:my] = 1
        else:
            gt_matrix_1[x:mx, y:my] = 1
    tp1, fp1, fn1, tn1 = 0, 0, 0, 0
    tp2, fp2, fn2, tn2 = 0, 0, 0, 0
    for j in range(n):
        for k in range(m):
            # Checking class 'with mask':
            if gt_matrix_0[j, k]:
                if a_matrix_0[j, k]:
                    tp1 += 1
                else:
                    fn1 += 1
            else:
                if a_matrix_0[j, k]:
                    fp1 += 1
                else:
                    tn1 += 1
            # Checking class 'without_mask'
            if gt_matrix_1[j, k]:
                if a_matrix_1[j, k]:
                    tp2 += 1
                else:
                    fn2 += 1
            else:
                if a_matrix_1[j, k]:
                    fp2 += 1
                else:
                    tn2 += 1
    pres1, pres2 = 0 if tp1 + fp1 == 0 else tp1 / (tp1 + fp1), 0 if tp2 + fp2 == 0 else tp2 / (tp2 + fp2)
    rec1, rec2 = 0 if tp1 + fn1 == 0 else tp1 / (tp1 + fn1), 0 if tp2 + fn2 == 0 else tp2 / (tp2 + fn2)
    p_1.append(pres1)
    p_2.append(pres2)
    r_1.append(rec1)
    r_2.append(rec2)

# %% Plotting results
fig, ax = plt.subplots()
ax.scatter(r_1, p_1)
ax.set(xlabel='Cobertura', ylabel='Precisión', title='Precisión vs. cobertura')
ax.grid()
fig.savefig('PvsC_1.png')
fig.show()

# %%
# f_1 = 2 * np.average(p_1) * np.average(r_1) / (np.average(p_1) + np.average(r_1))
nr2, np2 = np.asarray(r_2)[np.asarray(r_2) > 0], np.asarray(p_2)[np.asarray(p_2) > 0]
# f_2 = 2 * np.average(p_2) * np.average(r_2) / (np.average(p_2) + np.average(r_2))
f_2 = 2 * np.average(np2) * np.average(nr2) / (np.average(np2) + np.average(nr2))
fig, ax = plt.subplots()
ax.scatter(nr2, np2)
ax.set(xlabel='Cobertura', ylabel='Precisión', title='Precisión vs. cobertura')
ax.grid()
fig.savefig('PvsC_2.png')
fig.show()


