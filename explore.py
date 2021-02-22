# %% Imports
import random
import skimage.io as io
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# %% Displaying images and annotations
# Taken from https://docs.python.org/3/library/os.html
n = 4
selected_images = []
for root, dirs, files in os.walk(os.path.join('.', 'images')):
    selected_images = random.sample(files, n)
# Taken from https://docs.python.org/3/library/xml.etree.elementtree.html
selected_annotations = []
for i in range(n):
    tree = ET.parse(os.path.join('annotations', selected_images[i].split('.')[0] + '.xml'))
    root = tree.getroot()
    instances = ''
    for tags in root.findall('object'):
        instances += tags.find('name').text + ', '
    instances = instances[:-2]
    selected_annotations.append(instances)
# Image display
fig, axs = plt.subplots(n, 2)
for i in range(n):
    img = io.imread(os.path.join('images', selected_images[i]))
    axs[i, 0].imshow(img)
    axs[i, 0].axis('off')
    axs[i, 1].text(0.5, 0.5, selected_annotations[i], wrap=True, ha='center', fontsize=8)
    axs[i, 1].axis('off')
fig.suptitle('Imágenes y sus correspondientes instancias detectadas (anotaciones)')
fig.show()
fig.savefig('sample_database_images.png')
