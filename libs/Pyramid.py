import cv2 as cv


# Class taken and adjusted from VladKha's object detector: https://github.com/VladKha/object_detector
class Pyramid:
    def __init__(self, downscale, min_size_window):
        self.downscale = downscale
        self.min_size_window = min_size_window

    # Generates resized images (image pyramid) based on a specified downscale factor and
    # a minimum size window (both specified in parameters). images are yielded as they are created.
    def pyramid(self, image):
        # Yielding allows the method to give the resized images as they're created, rather than waiting
        # for the creation of an array that is then returned. First, the original image is given.
        yield image
        while True:
            # OpenCV receives the new dimension as (width, height).
            dim = (int(image.shape[1] / self.downscale), int(image.shape[0] / self.downscale))
            image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
            # If the new image's dimensions are smaller than allowed, the pyramid is finished.
            if image.shape[0] < self.min_size_window[0] or image.shape[1] < self.min_size_window[1]:
                break
            # Yields the resized image to be used on creation.
            yield image
