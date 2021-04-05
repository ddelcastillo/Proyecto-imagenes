# %% Imports
import random
import skimage.io as io
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from matplotlib.lines import Line2D

# %% Displaying images and annotations
# Taken from https://docs.python.org/3/library/os.html
n = 2  # Number of images to select (should be at least 2!).
if n < 2 or n % 1 != 0:
    raise Exception('Invalid number of images (n should be an integer greater or equal to 2).')
# Randomly selects n images form the database.
selected_images = []
for root, dirs, files in os.walk(os.path.join('.', 'images')):
    selected_images = random.sample(files, n)
# Taken from https://docs.python.org/3/library/xml.etree.elementtree.html
# Based on the randomly selected images, gets the instance information
# for each respective image XML file. Makes a string to later plot.
selected_annotations = []
for i in range(n):
    tree = ET.parse(os.path.join('annotations', selected_images[i].split('.')[0] + '.xml'))
    root = tree.getroot()
    instances = ''
    for tags in root.findall('object'):
        instances += tags.find('name').text + ', '
    instances = instances[:-2]
    selected_annotations.append(instances)

# %% Class instance exploration
# Counts all instances for each image for all the XML files. Stores them
# in a dictionary and then plots the frequencies in a bar graph.
count = {}
for root, dirs, files in os.walk(os.path.join('.', 'annotations')):
    for annotation in files:
        tree = ET.parse(os.path.join('annotations', annotation))
        root = tree.getroot()
        for tags in root.findall('object'):
            instance = tags.find('name').text
            if instance in count:
                count[instance] += 1
            else:
                count[instance] = 1

# %% Generating image annotation.
# If annotations should be written above the box.
write_annotations = False
# Annotations on figures might overflow depending on the image and the number of annotations.
# Since images are small, it's difficult not to have certain images overflowing, better to just resample.
fig, axs = plt.subplots(1, n)
for i in range(n):
    img = io.imread(os.path.join('images', selected_images[i]))
    tree = ET.parse(os.path.join('annotations', selected_images[i].split('.')[0] + '.xml'))
    root = tree.getroot()
    box_info = []
    for tags in root.findall('object'):
        name = tags.find('name').text
        box = tags.find('bndbox')
        xmin = max(0, int(box.find('xmin').text)) % img.shape[1]
        xmax = max(0, int(box.find('xmax').text)) % img.shape[1]
        ymin = max(0, int(box.find('ymin').text)) % img.shape[0]
        ymax = max(0, int(box.find('ymax').text)) % img.shape[0]
        k = list(count.keys()).index(name) % img.ndim
        # Draws a rectangle around the detected area. Color is based on index (kinda generic
        # but not much beyond 4 classes since it might start repeating colors; doesn't matter though).
        for j in range(xmin, xmax):
            img[ymin, j, k] = 255
            img[ymax, j, k] = 255
        for j in range(ymin, ymax):
            img[j, xmin, k] = 255
            img[j, xmax, k] = 255
        # Original dataset has a typo with 'weared' (should be 'worn').
        name = 'mask_worn_incorrectly' if name == 'mask_weared_incorrect' else name
        box_info.append({'ymin': ymin, 'xmin': xmin, 'xmax': xmax, 'name': name})
    axs[i].imshow(img)
    axs[i].axis('off')
    # Runs through each annotation's box and writes the annotation's text above it.
    if write_annotations:
        for j in box_info:
            axs[i].text(j['xmin'] + int((j['xmax'] - j['xmin']) / 2), j['ymin'], j['name'],
                        backgroundcolor='white', color='black', ha='center',
                        fontsize=4)
fig.suptitle('Imágenes y sus correspondientes instancias detectadas (anotaciones)')
# Taken from https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
custom_lines = []
for i in range(img.ndim):
    custom_lines.append(Line2D([0], [0], color='rgb'[i % 3], lw=4))
fig.legend(custom_lines, ['without_mask', 'with_mask', 'mask_worn_incorrectly'], loc='lower right')
fig.show()
fig.savefig('sample_database_images.png')
input('Press enter to continue...')

# %% Bar graph generation
# Bar graph generation, taken from https://matplotlib.org/3.3.4/api/_as_gen/matplotlib.pyplot.bar.html
# Hardcoded classes since there is a typo 'weared' (should be 'worn').
bars = plt.bar(['without_mask', 'with_mask', 'mask_worn_incorrectly'], list(count.values()))
xmin, xmax, ymin, ymax = plt.axis()
for bar in bars:
    y = bar.get_height()
    # Values are set manually for offset of values on top of the bar.
    off_x = 0.4
    off_y = int(0.2 * (ymax - max(count.values())))  # Value at 80% below top. For all bars.
    plt.text(bar.get_x() + off_x, y + off_y, y, ha='center')
plt.suptitle('Distribución del número de instancias (clases)')
plt.savefig('instance_count.png')
plt.show()
input('Press enter to finalize.')
