import argparse as ap
import os

import cv2
import matplotlib.pyplot as plt
import scipy.misc
from skimage.feature import hog
from skimage.color import rgb2gray
import joblib
import numpy as np

from utils import sliding_window, pyramid, non_max_suppression

WINDOW_SIZE = [178, 218]
WINDOW_STEP_SIZE = 20
ORIENTATIONS = 9
PIXELS_PER_CELL = [8, 8]
CELLS_PER_BLOCK = [3, 3]
VISUALISE = False
NORMALISE = None
THRESHOLD = 0.4
MODEL_PATH = 'hog_svm_model.joblib'
PYRAMID_DOWNSCALE = 1.5
POS_SAMPLES = 1000
NEG_SAMPLES = 1000
RANDOM_STATE = 42


class Detector:
    def __init__(self, downscale=1.5, window_size=(178, 218), window_step_size=32, threshold=0.4):
        self.clf = joblib.load(MODEL_PATH)
        self.downscale = downscale
        self.window_size = window_size
        self.window_step_size = window_step_size
        self.threshold = threshold

    def detect(self, image):
        clone = image.copy()
        image = rgb2gray(image)
        detections = []
        downscale_power = 0
        for im_scaled in pyramid(image, downscale=self.downscale, min_size=self.window_size):
            if im_scaled.shape[0] < self.window_size[1] or im_scaled.shape[1] < self.window_size[0]:
                break
            for (x, y, im_window) in sliding_window(im_scaled, self.window_step_size,
                                                    self.window_size):
                if im_window.shape[0] != self.window_size[1] or im_window.shape[1] != self.window_size[0]:
                    continue
                feature_vector = hog(im_window)
                X = np.array([feature_vector])
                prediction = self.clf.predict(X)
                if prediction == 1:
                    x1 = int(x * (self.downscale ** downscale_power))
                    y1 = int(y * (self.downscale ** downscale_power))
                    detections.append((x1, y1,
                                       x1 + int(self.window_size[0] * (
                                               self.downscale ** downscale_power)),
                                       y1 + int(self.window_size[1] * (
                                               self.downscale ** downscale_power))))
            downscale_power += 1
        clone_before_nms = clone.copy()
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(clone_before_nms, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        # Perform Non Maxima Suppression
        detections = non_max_suppression(np.array(detections), self.threshold)

        clone_after_nms = clone
        # Display the results after performing NMS
        for (x1, y1, x2, y2) in detections:
            # Draw the detections
            cv2.rectangle(clone_after_nms, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        return clone_before_nms, clone_after_nms


if __name__ == '__main__':
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-i', '--images_dir_path', help='Path to the test images dir',
                        required=True)
    parser.add_argument('-v', '--visualize', help='Visualize the sliding window',
                        action='store_true')
    args = vars(parser.parse_args())

    visualize_det = args['visualize']
    image_dir_path = args['images_dir_path']

    detector = Detector(downscale=PYRAMID_DOWNSCALE, window_size=WINDOW_SIZE,
                        window_step_size=WINDOW_STEP_SIZE, threshold=THRESHOLD)

    for image_name in os.listdir(image_dir_path):
        if image_name == '.DS_Store':
            continue

        # Read the image
        image = scipy.misc.imread(os.path.join(image_dir_path, image_name))

        # detect faces and return 2 images - before NMS and after
        image_before_nms, image_after_nms = detector.detect(image)

        # show image before NMS
        plt.imshow(image_before_nms)
        plt.xticks([]), plt.yticks([])
        plt.show()

        # show image after NMS
        plt.imshow(image_after_nms)
        plt.xticks([]), plt.yticks([])
        plt.show()
