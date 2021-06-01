import cv2
import numpy as np


def pyramid(image, downscale=1.5, min_size=(30, 30)):
    yield image
    while True:
        w = int(image.shape[1] / downscale)
        image = resize(image, width=w)
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image


def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield x, y, image[y:y + window_size[1], x:x + window_size[0]]


# Malisiewicz et al.
def non_max_suppression(boxes, overlap_thresh=0.7):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indexes = np.argsort(y2)
    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[indexes[:last]])
        yy1 = np.maximum(y1[i], y1[indexes[:last]])
        xx2 = np.minimum(x2[i], x2[indexes[:last]])
        yy2 = np.minimum(y2[i], y2[indexes[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[indexes[:last]]
        indexes = np.delete(indexes, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))
    return boxes[pick].astype("int")


def bb_intersection(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    t1 = x2 - x1 + 1
    t2 = y2 - y1 + 1
    if t1 <= 0 or t2 <= 0:
        intersection_area = 0
    else:
        intersection_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    return intersection_area


def bb_intersection_over_union(box_a, box_b):
    intersection_area = bb_intersection(box_a, box_b)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    iou = intersection_area / (box_a_area + box_b_area - intersection_area)
    return iou


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized