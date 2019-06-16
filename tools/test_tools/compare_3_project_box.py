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


xoy_image_folder = "/Users/jiangyy/projects/medical-rib/data/voc2007_rawbox/JPEGImages"
xoy_xml_folder = "/Users/jiangyy/projects/medical-rib/data/voc2007_rawbox/Annotations"
yoz_image_folder = "/Users/jiangyy/projects/medical-rib/data/voc2007_rawbox_yoz/JPEGImages"
yoz_xml_folder = "/Users/jiangyy/projects/medical-rib/data/voc2007_rawbox_yoz/Annotations"
xoz_image_folder = "/Users/jiangyy/projects/medical-rib/data/voc2007_rawbox_xoz/JPEGImages"
xoz_xml_folder = "/Users/jiangyy/projects/medical-rib/data/voc2007_rawbox_xoz/Annotations"

#more_image_folder = few_image_folder
#more_xml_folder = few_xml_folder
share_df = pd.read_csv("../problems_for_labels/share_box_ribs.csv")

for file in os.listdir(xoy_image_folder):
    id = file[0:-4]
#for id in share_df['jpeg_name'].unique():

    #if not id.startswith("135402000404094"):
    #    continue
    #if not id.__contains__('135402000612792-2800-R4_R3'):
    #    continue
    print(id)
    xoy_image_path = "{}/{}.jpg".format(xoy_image_folder, id)
    xoy_xml_path = "{}/{}.xml".format(xoy_xml_folder, id)
    yoz_image_path = "{}/{}.jpg".format(yoz_image_folder, id)
    yoz_xml_path = "{}/{}.xml".format(yoz_xml_folder, id)
    xoz_image_path = "{}/{}.jpg".format(xoz_image_folder, id)
    xoz_xml_path = "{}/{}.xml".format(xoz_xml_folder, id)

    tags = id.split('-')[-1].split('_')

    if not(os.path.exists(xoy_image_path) and os.path.exists(xoy_xml_path)):
        print("xoy file un")
        continue

    if not(os.path.exists(yoz_image_path) and os.path.exists(yoz_xml_path)):
        print("yoz file un")
        continue

    if not(os.path.exists(xoz_image_path) and os.path.exists(xoz_xml_path)):
        print("xoz file un")
        continue

    plt.figure(figsize=(12, 4))
    grid = plt.GridSpec(4, 12, wspace=0.5, hspace=0.5)

    image_path, xml_path, tag = xoy_image_path, xoy_xml_path, "xoy"
    plt.subplot(grid[0:4, 0:4])
    image = Image.open(image_path)
    plt.imshow(image)
    objs, obj_wh = parse_rec(xml_path)
    print("old:",obj_wh)
    for obj in objs:
        truth = obj['bbox']
        plt.gca().add_patch(patches.Rectangle((truth[0], truth[1]), (truth[2] - truth[0]), (truth[3] - truth[1]),
                                              linewidth=1, edgecolor='r', facecolor='none'))

    image_path, xml_path, tag = yoz_image_path, yoz_xml_path, "yoz"
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

    image_path, xml_path, tag = xoz_image_path, xoz_xml_path, "yoz"
    plt.subplot(grid[0:4, 8:12])
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
    ax = plt.gca()
    ax.invert_yaxis()
    plt.xlabel(id)
    #plt.savefig("/Users/jiangyy/projects/medical-rib/data/raw_box/{}.png".format(id))
    plt.show()
