import numpy as np
from skimage.exposure import equalize_hist
from skimage.transform import resize


class Classifier:
    def __init__(self, model, window_size):
        self.model = model
        self.window_size = window_size

    def classify_detections(self, detections):
        images = np.zeros([len(detections), self.window_size[0] * self.window_size[1]])
        for index, image in enumerate(images):
            images[index, :] = resize(equalize_hist(image), (self.window_size[0], self.window_size[1])).flatten()
        return self.model.predict(images)
