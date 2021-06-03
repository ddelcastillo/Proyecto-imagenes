import argparse
import os
import joblib
import xml.etree.ElementTree as ET
import numpy as np
import cv2 as cv
import skimage.io as io
import pickle
import time
import warnings
from tqdm import tqdm
from sklearn.metrics import classification_report
from libs.detector import Detector
from libs.classifier import Classifier
from libs.parameters import *

warnings.filterwarnings("ignore")

MODEL_FOLDER = 'modelos'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None, help='Nombre de la imagen que se va a evaluar')
    args = parser.parse_args()
    # Loading models
    hog_svm_model = joblib.load(os.path.join(MODEL_FOLDER, DETECTOR_MODEL_NAME))
    classifier_rf_model = joblib.load(os.path.join(MODEL_FOLDER, CLASSIFIER_MODEL_NAME))
    detector = Detector(hog_svm_model, DOWNSCALE, MIN_SIZE_WINDOW, WINDOW_SIZE, WIDTH_STEP, HEIGHT_STEP, THRESHOLD)
    classifier = Classifier(classifier_rf_model, WINDOW_SIZE)
    if args.image:
        bar = tqdm(total=1)
        image = io.imread(os.path.join('Datos', 'Imagenes', 'dataset', args.image))
        detections = detector.detect(image)
        results = classifier.classify_detections(detections)
        bar.update(1)
        time.sleep(1)
        tree = ET.parse(os.path.join('Datos', 'Anotaciones', args.image.split('.')[0] + '.xml'))
        root = tree.getroot()
        total_detections = 0
        total_mask = 0
        for tags in root.findall('object'):
            total_detections += 1
            if tags.find('name').text == 'with_mask':
                total_mask += 1
        print(f'\nDetected {len(detections)}/{total_detections} faces.')
        if total_detections < len(detections):
            total_detections = len(detections)
        y_real = np.concatenate(
            (np.ones(total_mask, dtype=np.int_), np.zeros(total_detections - total_mask, dtype=np.int_)))
        y_hat = np.concatenate((sorted(results), np.zeros(len(y_real)-len(results), dtype=np.int_)))
        print(classification_report(y_real, y_hat, target_names=['With mask', 'Without mask']))
    else:
        image_list = os.listdir(os.path.join('Datos', 'Imagenes', 'dataset'))
        big_y_real = np.array([])
        big_y_hat = np.array([])
        bar = tqdm(total=len(image_list))
        for image_name in image_list:
            image = io.imread(os.path.join('Datos', 'Imagenes', 'dataset', image_name))
            detections = detector.detect(image)
            if len(detections) == 0:
                continue
            results = classifier.classify_detections(detections)
            bar.update(1)
            tree = ET.parse(os.path.join('Datos', 'Anotaciones', image_name.split('.')[0] + '.xml'))
            root = tree.getroot()
            total_detections = 0
            total_mask = 0
            for tags in root.findall('object'):
                total_detections += 1
                if tags.find('name').text == 'with_mask':
                    total_mask += 1
            if total_detections < len(detections):
                total_detections = len(detections)
            y_real = np.concatenate(
                (np.ones(total_mask, dtype=np.int_), np.zeros(total_detections - total_mask, dtype=np.int_)))
            y_hat = np.concatenate((sorted(results), np.zeros(len(y_real) - len(results), dtype=np.int_)))
            big_y_real = np.concatenate((big_y_real, y_real))
            big_y_hat = np.concatenate((big_y_hat, y_hat))
        time.sleep(1)
        print(classification_report(big_y_real, big_y_hat, target_names=['With mask', 'Without mask']))
