import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
import numpy as np
import xml.etree.ElementTree as ET
import os


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    size = tree.find('size')
    width, height = int(size.find('width').text), int(size.find('height').text)

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

    return objects, {'width': width, 'height': height}


few_image_folder = "/Users/jiangyy/projects/medical-rib/data/voc2007/JPEGImages"
few_xml_folder = "/Users/jiangyy/projects/medical-rib/data/voc2007/Annotations"
more_image_folder = "/Users/jiangyy/projects/medical-rib/data/voc2007.refined/JPEGImages" #"/Users/jiangyy/voc2007.xoy/JPEGImages"
more_xml_folder = "/Users/jiangyy/projects/medical-rib/data/voc2007.refined/Annotations.refined"

#more_image_folder = few_image_folder
#more_xml_folder = few_xml_folder

for file in os.listdir(few_xml_folder):
    id = file[0:-4]
    #if not id.startswith("135402000404094"):
    #    continue
    #if not id.__contains__('135402000612792-2800-R4_R3'):
    #    continue
    print(id)
    few_image_path = "{}/{}.jpg".format(few_image_folder, id)
    few_xml_path = "{}/{}.xml".format(few_xml_folder, id)
    more_image_path = "{}/{}.jpg".format(more_image_folder, id)
    more_xml_path = "{}/{}.xml".format(more_xml_folder, id)

    tags = id.split('-')[-1].split('_')

    if not(os.path.exists(few_image_path) and os.path.exists(few_xml_path)):
        print("few file un")
        continue

    if not(os.path.exists(more_image_path) and os.path.exists(more_xml_path)):
        print("more file un")
        continue

    plt.figure(figsize=(12, 4))
    grid = plt.GridSpec(4, 12, wspace=0.5, hspace=0.5)

    image_path, xml_path, tag = few_image_path, few_xml_path, "few"
    plt.subplot(grid[0:4, 0:4])
    image = Image.open(image_path)
    plt.imshow(image)
    objs, obj_wh = parse_rec(xml_path)
    print("old:",obj_wh)
    for obj in objs:
        truth = obj['bbox']
        plt.gca().add_patch(patches.Rectangle((truth[0], truth[1]), (truth[2] - truth[0]), (truth[3] - truth[1]),
                                              linewidth=1, edgecolor='r', facecolor='none'))

    image_path, xml_path, tag = more_image_path, more_xml_path, "more"
    plt.subplot(grid[0:4, 4:8])
    image = Image.open(image_path)
    plt.imshow(image)
    objs, obj_wh = parse_rec(xml_path)
    print("new:", obj_wh)
    for obj in objs:
        truth = obj['bbox']
        plt.gca().add_patch(patches.Rectangle((truth[0], truth[1]), (truth[2] - truth[0]), (truth[3] - truth[1]),
                                              linewidth=1, edgecolor='g', facecolor='none'))
        if obj['difficult']:
            plt.text(truth[2], truth[3], "diff", bbox=dict(facecolor='red', alpha=0.5))

    plt.subplot(grid[0:4, 8:12])
    plt.xlim(-800, 800)
    plt.ylim(-800, 800)
    objs, obj_wh = parse_rec(few_xml_path)

    for obj in objs:
        truth = obj['bbox']
        plt.gca().add_patch(patches.Rectangle((truth[0], truth[1]), (truth[2] - truth[0]), (truth[3] - truth[1]),
                                              linewidth=1, edgecolor='r', facecolor='none'))
    objs, obj_wh = parse_rec(more_xml_path)
    for obj in objs:
        truth = obj['bbox']
        plt.gca().add_patch(patches.Rectangle((truth[0], truth[1]), (truth[2] - truth[0]), (truth[3] - truth[1]),
                                              linewidth=1, edgecolor='g', facecolor='none'))

    ax = plt.gca()
    ax.invert_yaxis()
    plt.xlabel("_".join(tags))
    plt.show()
