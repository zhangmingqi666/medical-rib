# -*- coding:utf-8-*-
import os
import argparse
import sys
import pandas as pd
import warnings
import datetime
import json
from pycococreatortools import pycococreatortools
from PIL import Image
warnings.filterwarnings('ignore')


_ROOT_NODE = 'annotation'
# folder
_FOLDER_NODE = 'hehehe'
# filename
_FILE_NAME = '000001.jpg'
# source
_DATABASE_NAME = 'rib-fracture-detection'
_ANNOTATION = 'rib 2018'
_IMAGE_SOURCE = 'hua-dong hospital'
# owner
_AUTHOR = 'jiangyy,leaves'
_SEGMENTED = '0'
# object
_NAME = 'person'
_DIFFICULT = '0'
_TRUNCATED = '0'
_POSE = 'Unspecified'

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'fragment',
        'supercategory': 'hurt'
    }
]


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info


def create_annotation_info(annotation_id, image_id, category_info, bounding_box,
                           image_size=None, tolerance=2):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": (bounding_box[2]*bounding_box[3]).__int__(),
        "bbox": bounding_box,
        "segmentation": 0,
        "width": image_size[0],
        "height": image_size[1],
    }

    return annotation_info


def generate_coco(df, coco_json_file=None):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    # id,location_id,x.max,x.min,y.max,y.min,z.max,z.min,type,dataSet_id,length_x,offset_x,length_y,offset_y,length_z,offset_z
    for index, row in df.iterrows():
        dataSet_id = row['dataSet_id']
        image_path = "{}/{}.jpg".format("/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/before_augmented",dataSet_id)
        image_size = row['length_y'], row['length_x']
        image_info = create_image_info(
            image_id, os.path.basename(image_path), image_size)
        coco_output["images"].append(image_info)

        bounding_box = [row['y.min'], row['x.min'], row['y.max']-row['y.min'], row['x.max']-row['x.min']]

        # class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
        category_info = {'id': 1, 'is_crowd': 0}

        annotation_info = create_annotation_info(
            segmentation_id, image_id, category_info, bounding_box,
            image_size, tolerance=2)

        coco_output["annotations"].append(annotation_info)
        segmentation_id = segmentation_id + 1
        image_id = image_id + 1

    with open('instances_shape_train2018.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('--label_loc_type_info_path', required=True, dest='label_loc_type_info_path',
                        action='store', help='label_loc_type_info_path')
    parser.add_argument('--coco_json_file', required=True, dest='coco_json_file',
                        action='store', help='coco_json_file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    assert os.path.exists(args.label_loc_type_info_path)
    df = pd.read_csv(args.label_loc_type_info_path)

    generate_coco(df, coco_json_file=args.coco_json_file)






