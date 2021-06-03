# Class taken and adjusted from VladKha's object detector: https://github.com/VladKha/object_detector
# Will return multiple sliding windows based on it's configured window size.
class SlidingWindow:
    def __init__(self, window_size, width_step, height_step):
        self.window_size = window_size
        self.width_step = width_step
        self.height_step = height_step

    # Generates windows obtained from sliding a specified window size across the image.
    # It advances based on specific height and width steps (specified in parameters).
    def generate_sliding_window(self, image):
        for y in range(0, image.shape[0], self.height_step):
            for x in range(0, image.shape[1], self.width_step):
                yield x, y, image[y:(y+self.window_size[1]), x:(x+self.window_size[0])]
