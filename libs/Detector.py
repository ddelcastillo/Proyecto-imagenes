import sys
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from .pyramid import Pyramid
from .slidingWindow import SlidingWindow
from .nonMaxSuppressor import NonMaxSuppressor

sys.path.append('../')

# Class taken and adjusted from VladKha's object detector: https://github.com/VladKha/object_detector
# Class that will returns detections for an image based on the given configurations.
class Detector:
    def __init__(self, model, downscale, min_size_window, window_size, width_step, height_step, threshold):
        self.model = model
        self.downscale = downscale
        self.min_size_window = min_size_window
        self.window_size = window_size
        self.width_step = width_step
        self.height_step = height_step
        self.threshold = threshold

    # Generates an array of detections based on the given model and configurations. Assumes that
    # a class is detected if the prediction returns 1. Uses an image pyramid and non-max suppression.
    def detect(self, image):
        gray_image, detections, downscale_power = rgb2gray(image), [], 0
        pyramid = Pyramid(self.downscale, self.min_size_window)
        for image_scaled in pyramid.pyramid(gray_image):
            window = SlidingWindow(self.window_size, self.width_step, self.height_step)
            for (x, y, image_window) in window.generate_sliding_window(image_scaled):
                if image_window.shape[0] != self.window_size[0] or image_window.shape[1] != self.window_size[1]:
                    continue
                hog_features = np.array([hog(image_window)])
                prediction = self.model.predict(hog_features)
                if prediction == 1:
                    x1 = int(x * (self.downscale ** downscale_power))
                    y1 = int(y * (self.downscale ** downscale_power))
                    detections.append((x1, y1,
                                       x1 + int(self.window_size[0] * (
                                               self.downscale ** downscale_power)),
                                       y1 + int(self.window_size[1] * (
                                               self.downscale ** downscale_power))))
            downscale_power += 1
        non_max_suppressor = NonMaxSuppressor(threshold=self.threshold)
        return non_max_suppressor.apply_non_max_suppression(np.array(detections))
