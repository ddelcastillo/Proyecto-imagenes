# %% Imports
import random
import skimage.io as io
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# %% Displaying images and annotations
# Taken from https://docs.python.org/3/library/os.html
n = 4   # Number of images to select
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
# Displays the image (left) with it's respective list of detected instances
# (annotations, right). Displays and shows the image.
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
input('Press enter to continue...')

# %% Class instance exploration

# Counts all instances for each image in for all the XML files. Stores them
# in a dictionary and them plots the frequencies in a bar graph.
count = {}
for root, dirs, files in os.walk(os.path.join('.', 'annotations')):
    for annotation in files:
        tree = ET.parse(os.path.join('annotations',annotation))
        root = tree.getroot()
        for tags in root.findall('object'):
            instance = tags.find('name').text
            if instance in count:
                count[instance] += 1
            else:
                count[instance] = 1
# Bar graph generation
fig, ax = plt.subplots(1)
ax.bar(list(count.keys()), list(count.values()))
fig.suptitle('Distribución del número de instancias (clases)')
fig.show()
fig.savefig('instance_count.png')
input('Press enter to finalize.')
