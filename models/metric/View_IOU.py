import xml.etree.ElementTree as ET
import os
#import cPickle
import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def read_det(filename):
    # read dets
    with open(filename, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    return image_ids, confidence, BB

image_ids, confidence, BB = read_det("/Users/jiangyy/projects/medical-rib/models/darknet/results/hurt.txt")
for id, p, box in zip(image_ids, confidence, BB):
    groudtruth_path = "/Users/jiangyy/voc2007.xoy/Annotations/{}.xml".format(id)
    image_path = "/Users/jiangyy/voc2007.xoy/JPEGImages/{}.jpg".format(id)
    obj = parse_rec(groudtruth_path)
    truth = obj[0]['bbox']
    fig, ax = plt.subplots(1)
    image = Image.open(image_path)
    ax.imshow(image)
    # box = [600, 86, 610, 109]
    rect = patches.Rectangle((box[0], box[1]), (box[2] - box[0]), (box[3] - box[1]), linewidth=1, edgecolor='r',
                             facecolor='none')
    rect1 = patches.Rectangle((truth[0], truth[1]), (truth[2] - truth[0]), (truth[3] - truth[1]), linewidth=1, edgecolor='g',
                             facecolor='none')
    ax.add_patch(rect)
    ax.add_patch(rect1)
    plt.show()

