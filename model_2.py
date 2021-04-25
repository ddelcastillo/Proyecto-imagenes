#%% Packages
import os
import skimage.io as io
from skimage import data
from skimage.feature import Cascade

import matplotlib.pyplot as plt
from matplotlib import patches

#%% Files
trained_file = data.lbp_frontal_face_cascade_filename()
detector = Cascade(trained_file)
img = io.imread(os.path.join('.', 'images', 'maksssksksss94.png'))
detected = detector.detect_multi_scale(img=img,
                                       scale_factor=1.2,
                                       step_ratio=1,
                                       min_size=(60, 60),
                                       max_size=(123, 123))
plt.imshow(img)
img_desc = plt.gca()
plt.set_cmap('gray')
for patch in detected:

    img_desc.add_patch(
        patches.Rectangle(
            (patch['c'], patch['r']),
            patch['width'],
            patch['height'],
            fill=False,
            color='r',
            linewidth=2
        )
    )
plt.show()
