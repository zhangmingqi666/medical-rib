
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
import numpy as np
import xml.etree.ElementTree as ET

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


ignore_xml_df = pd.read_csv("../../../experiments/label_data/ignore.csv")
big_box_df = pd.read_csv("../../../experiments/label_data/big_box.csv")
label_df = ignore_xml_df.append(big_box_df)
label_df['dataSet_id'] = label_df['xml_name'].apply(lambda x: x[:-4])

print("dataSet number :{}".format(len(label_df)))
join_label_df = pd.read_csv("../../../data/csv_files/join_label.csv")


object_label_df = label_df.merge(join_label_df, on='dataSet_id', how='left')
print(object_label_df[['dataSet_id', 'location_id']])

for idx, row in object_label_df.iterrows():
    dataset_id, location_id = row['dataSet_id'], row['location_id']
    x_min, x_max = row['range.x.min'], row['range.x.max']
    y_min, y_max = row['range.y.min'], row['range.y.max']
    z_min, z_max = row['range.z.min'], row['range.z.max']

    print("id:{}.xml".format(dataset_id))

    groudtruth_path = "/Users/jiangyy/voc2007.xoy/Annotations/{}.xml".format(dataset_id)
    image_path = "/Users/jiangyy/voc2007.xoy/JPEGImages/{}.jpg".format(dataset_id)
    revisedtruth_path = "/Users/jiangyy/voc2007.xoy/Annotations2/{}.xml".format(dataset_id)
    image = Image.open(image_path)
    nii_csv_path = "/Users/jiangyy/projects/medical-rib/data/nii_csv_files/{}.csv".format(location_id)

    loc_df = pd.read_csv(nii_csv_path)
    xoy_loc_df = loc_df.groupby(['x', 'y']).agg({'z': {'sum'}})
    xoy_loc_df.reset_index(inplace=True)
    x_point, y_point = xoy_loc_df['x'], xoy_loc_df['y']

    plt.figure(figsize=(8, 4))
    grid = plt.GridSpec(4, 8, wspace=0.5, hspace=0.5)
    plt.subplot(grid[0:4, 0:4])
    plt.imshow(image)
    obj = parse_rec(groudtruth_path)
    truth = obj[0]['bbox']
    plt.gca().add_patch(patches.Rectangle((truth[0], truth[1]), (truth[2] - truth[0]), (truth[3] - truth[1]),
                                          linewidth=1, edgecolor='r', facecolor='none'))

    obj = parse_rec(revisedtruth_path)
    for e in obj:
        revised_truth = e['bbox']
        plt.gca().add_patch(patches.Rectangle((revised_truth[0], revised_truth[1]), (revised_truth[2] - revised_truth[0]),
                                              (revised_truth[3] - revised_truth[1]),
                                              linewidth=1, edgecolor='g', facecolor='none'))

    plt.subplot(grid[0:4, 4:8])

    shape = np.array(Image.open(image_path)).shape
    plt.xlim(0, shape[1])
    plt.ylim(0, shape[0])
    plt.gca().invert_yaxis()
    plt.scatter(y_point - y_min, x_point - x_min, s=20, color='green')
    plt.gca().add_patch(patches.Rectangle((truth[0], truth[1]), (truth[2] - truth[0]), (truth[3] - truth[1]),
                                          linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()

