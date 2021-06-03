import numpy as np


# Class taken and adjusted from VladKha's object detector: https://github.com/VladKha/object_detector
# Will apply non-maximum suppression to avoid multiple overlapping detections based on a threshold value.
class NonMaxSuppressor:
    def __init__(self, threshold):
        self.threshold = threshold

    def apply_non_max_suppression(self, boxes):
        # if there are no boxes, returns an empty list.
        if len(boxes) == 0:
            return []
        if boxes.dtype.kind == "i":
            boxes = boxes.astype(np.float_)
        # Gets the box corner coordinates for all boxes (detections). The pick array
        # will contain all the detections that are picked such that their overlap is optimal.
        pick, x1, y1, x2, y2 = [], boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        area, indexes = (x2 - x1 + 1) * (y2 - y1 + 1), np.argsort(y2)
        while len(indexes) > 0:
            last = len(indexes) - 1
            i = indexes[last]
            # Picks the last detection and ads it to the pick list.
            pick.append(i)
            # Coordinates for the box that contains both boxes being compared (for IoU).
            xx1 = np.maximum(x1[i], x1[indexes[:last]])
            yy1 = np.maximum(y1[i], y1[indexes[:last]])
            xx2 = np.minimum(x2[i], x2[indexes[:last]])
            yy2 = np.minimum(y2[i], y2[indexes[:last]])
            w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)
            # Overlap defined as IoU, intersection is the box between each other, while
            # the union will be the area of the last box (the one being currently picked).
            overlap = (w * h) / area[indexes[:last]]
            # If the overlap (IoU) is larger than the threshold, it's removed as a pick.
            indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > self.threshold)[0])))
        return boxes[pick].astype(np.int_)
