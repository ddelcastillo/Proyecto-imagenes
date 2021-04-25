# %% Packages
import cv2 as cv
import random
import os

# %% Constants
image_list = os.listdir(os.path.join('images'))
lbp_cascade_frontal_face_file = 'lbpcascade_frontalface_improved.xml'
haarcascade_frontalface_default_file = 'haarcascade_frontalface_default.xml'

# %% Test
image_file_test = random.sample(image_list, 1)[0]
image_test = cv.imread(os.path.join('images', image_file_test))
image_test_gray = cv.cvtColor(image_test, cv.IMREAD_GRAYSCALE)
# face_cascade = cv.CascadeClassifier(lbp_cascade_frontal_face_file)
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
