#coding=utf-8
from generate_xml_voc2007 import *
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
import os

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


for f in os.listdir("/Users/jiangyy/voc2007.xoy/Annotations/"):
    print(f)
    id = f.replace('.xml', '')
    groudtruth_path = "/Users/jiangyy/voc2007.xoy/Annotations/{}.xml".format(id)
    image_path = "/Users/jiangyy/voc2007.xoy/JPEGImages/{}.jpg".format(id)
    revisedtruth_path = "/Users/jiangyy/voc2007.xoy/Annotations2/{}.xml".format(id)
    obj = parse_rec(groudtruth_path)
    truth = obj[0]['bbox']
    fig, ax = plt.subplots(1)
    image = Image.open(image_path)
    ax.imshow(image)
    # # box = [600, 86, 610, 109]
    # rect = patches.Rectangle((box[0], box[1]), (box[2] - box[0]), (box[3] - box[1]), linewidth=1, edgecolor='r',
    #                          facecolor='none')
    print(id, "old box:", truth)
    rect1 = patches.Rectangle((truth[0], truth[1]), (truth[2] - truth[0]), (truth[3] - truth[1]), linewidth=1, edgecolor='g',
                             facecolor='none')
    ax.add_patch(rect1)
    if os.path.exists(revisedtruth_path):
        obj = parse_rec(revisedtruth_path)
        box = obj[0]['bbox']
        rect2 = patches.Rectangle((box[0], box[1]), (box[2] - box[0]), (box[3] - box[1]), linewidth=1, edgecolor='r',
                                  facecolor='none')
        print("new box:", box)
        ax.add_patch(rect2)
    plt.show()

