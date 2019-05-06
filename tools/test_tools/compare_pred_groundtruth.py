
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


pred_df = pd.read_csv("../../models/darknet/results/comp4_det_test_hurt.txt", header=None, delimiter=' ')
pred_df.columns = ['file_id', 'prob', 'x_min', 'y_min', 'x_max', 'y_max']
# print(len(pred_df))

ground_truth_df = pd.read_csv("../../data/voc2007.refined/out.file", header=None)
ground_truth_df.columns = ['file_id']
# print(len(ground_truth_df))
# ground_truth_df['id1'] = ground_truth_df['file_name'].apply(lambda x: '-'.join(x[0:-4].split('-')[0:-1]))
# print(ground_truth_df['id1'])
merge_pred_ground_df = ground_truth_df.merge(pred_df, on='file_id', how='outer')
diff_df = merge_pred_ground_df[merge_pred_ground_df['prob'].isnull()]

for _id in np.unique(pred_df['file_id']):

    plt.figure(figsize=(4, 4))
    grid = plt.GridSpec(4, 4, wspace=0.5, hspace=0.5)

    image_path = '{}/{}.jpg'.format("/Users/jiangyy/projects/medical-rib/data/voc2007.test_unpredict/JPEGImages", _id)
    xml_path = '{}/{}.xml'.format("/Users/jiangyy/projects/medical-rib/data/voc2007.test_unpredict/Annotations", _id)
    plt.subplot(grid[0:4, 0:4])
    image = Image.open(image_path)
    plt.imshow(image)
    objs, obj_wh = parse_rec(xml_path)
    print("old:", obj_wh)
    for obj in objs:
        truth = obj['bbox']
        plt.gca().add_patch(patches.Rectangle((truth[0], truth[1]), (truth[2] - truth[0]), (truth[3] - truth[1]),
                                              linewidth=1, edgecolor='r', facecolor='none'))
    pred_locations = pred_df[pred_df['file_id'] == _id]
    if len(pred_locations) > 0:
        for _, row in pred_locations.iterrows():
            pred_box = row[['x_min', 'y_min', 'x_max', 'y_max']]
            plt.gca().add_patch(patches.Rectangle((pred_box[0], pred_box[1]), (pred_box[2] - pred_box[0]),
                                                  (pred_box[3] - pred_box[1]),
                                                  linewidth=1, edgecolor='g', facecolor='none'))

    plt.show()


