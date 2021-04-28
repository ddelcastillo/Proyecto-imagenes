# %% Imports
import random
import skimage.io as io
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from matplotlib.lines import Line2D

# %% Displaying images and annotations
# Taken from https://docs.python.org/3/library/os.html
n = 4  # Number of images to select (should be at least 2!).
if n < 2 or n % 1 != 0:
    raise Exception('Invalid number of images (n should be an integer greater or equal to 2).')
# Randomly selects n images form the database.
selected_images = []
for root, dirs, files in os.walk(os.path.join('images')):
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
total_instances = {}
for root, dirs, files in os.walk(os.path.join('annotations')):
    for annotation in files:
        tree = ET.parse(os.path.join('annotations', annotation))
        root = tree.getroot()
        c = 0
        for tags in root.findall('object'):
            c += 1
            instance = tags.find('name').text
            if instance in count:
                count[instance] += 1
            else:
                count[instance] = 1
        if c != 0:
            if c in total_instances:
                total_instances[c] += 1
            else:
                total_instances[c] = 1

# Hardcoded name of classes in Spanish for further graphs.
es_keys = list(count.keys())
es_keys[es_keys.index('without_mask')] = 'Sin máscara'
es_keys[es_keys.index('with_mask')] = 'Con máscara'
es_keys[es_keys.index('mask_weared_incorrect')] = 'Máscara mal usada'

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
        xmin = min(int(box.find('xmin').text), img.shape[1]-1)
        xmax = min(int(box.find('xmax').text), img.shape[1]-1)
        ymin = min(int(box.find('ymin').text), img.shape[0]-1)
        ymax = min(int(box.find('ymax').text), img.shape[0]-1)
        k = list(count.keys()).index(name) % img.ndim
        # Draws a rectangle around the detected area. Color is based on index (kinda generic
        # but not much beyond 4 classes since it might start repeating colors; doesn't matter though).
        thickness = 2  # Border image thickness
        for j in range(xmin, xmax):
            for d in range(thickness):
                img[min(ymin+d, img.shape[0]-1), j, k] = 255
                img[min(ymax+d, img.shape[0]-1), j, k] = 255
                for e in range(1, img.ndim):
                    img[min(ymin+d, img.shape[0]-1), j, (k+e) % img.ndim] = 0
                    img[min(ymax+d, img.shape[0]-1), j, (k+e) % img.ndim] = 0
        for j in range(ymin, ymax):
            for d in range(thickness):
                img[j, min(xmin+d, img.shape[1]-1), k] = 255
                img[j, min(xmax+d, img.shape[1]-1), k] = 255
                for e in range(1, img.ndim):
                    img[j, min(xmin+d, img.shape[1]-1), (k+e) % img.ndim] = 0
                    img[j, min(xmax+d, img.shape[1]-1), (k+e) % img.ndim] = 0
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
fig.legend(custom_lines, es_keys, loc='lower right')
fig.show()
fig.savefig('sample_database_images.png')
input('Press enter to continue...')

# %% Bar graph generation
# Bar graph generation, taken from https://matplotlib.org/3.3.4/api/_as_gen/matplotlib.pyplot.bar.html
# Hardcoded classes since there is a typo 'weared' (should be 'worn').
bars = plt.bar(es_keys, list(count.values()))
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
input('Press enter to continue...')

# %% Number of annotations per image.
instances = dict(sorted(total_instances.items()))
bars = plt.bar(list(instances.keys()), list(instances.values()))
xmin, xmax, ymin, ymax = plt.axis()
plt.suptitle('Número de imágenes en función del número de instancias (por imagen)')
plt.xlim([0, max(total_instances.keys())+1])
plt.savefig('total_instances.png')
plt.show()
input('Press enter to finalize continue...')

# %% Pie chart generation
# Max_count is a threshold value used so that the pie chart has visible and readable
# values, since if all instances are included, image will overflow and become incomprehensible.
max_count = 10
max_count_val = 0
mod_instances = {}
for i in instances.keys():
    if i < max_count:
        mod_instances[i] = instances[i]
    else:
        max_count_val += instances[i]
mod_instances[str(max_count)+'+'] = max_count_val
fig, axs = plt.subplots()
# Taken from https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html
axs.pie(list(mod_instances.values()), labels=[str(i) for i in list(mod_instances.keys())],
        autopct='%1.1f%%', startangle=90)
fig.suptitle('Porcentaje del número de total imágenes en función\n del número de instancias (por imagen)')
fig.show()
fig.savefig('total_instances_pie.png')
input('Press enter to finalize.')
