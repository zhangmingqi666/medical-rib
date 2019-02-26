# -*- coding:utf-8-*-

import xml.dom
import xml.dom.minidom
import os
import argparse
import sys
import pandas as pd
import warnings
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


def create_element_node(doc, tag, attr):
    element_node = doc.createElement(tag)
    text_node = doc.createTextNode(attr)
    element_node.appendChild(text_node)
    return element_node


def create_child_node(doc, tag, attr, parent_node):
    child_node = create_element_node(doc, tag, attr)
    parent_node.appendChild(child_node)


def create_object_node(doc, attrs):
    _object_node = doc.createElement('object')

    create_child_node(doc, 'name', attrs.get('name', _NAME), _object_node)
    create_child_node(doc, 'pose', attrs.get('object.pose', _POSE), _object_node)
    create_child_node(doc, 'truncated', attrs.get('object.truncated', _TRUNCATED), _object_node)
    create_child_node(doc, 'difficult', attrs.get('object.difficult', _DIFFICULT), _object_node)

    _bndbox_list = attrs.get('bndbox', [0, 0, 0, 0])
    _bndbox_node = doc.createElement('bndbox')
    create_child_node(doc, 'xmin', str(_bndbox_list[0]), _bndbox_node)
    create_child_node(doc, 'ymin', str(_bndbox_list[1]), _bndbox_node)
    create_child_node(doc, 'xmax', str(int(_bndbox_list[0] + _bndbox_list[2])), _bndbox_node)
    create_child_node(doc, 'ymax', str(int(_bndbox_list[1] + _bndbox_list[3])), _bndbox_node)
    _object_node.appendChild(_bndbox_node)

    return _object_node


def create_annotation_node(attrs={}):
    """create the related xml format"""
    my_dom = xml.dom.getDOMImplementation()
    doc = my_dom.createDocument(None, _ROOT_NODE, None)
    root_node = doc.documentElement
    create_child_node(doc, 'folder', attrs.get('folder', _FOLDER_NODE), root_node)
    # print(attrs.get('filename', _FILE_NAME))
    create_child_node(doc, 'filename', attrs.get('filename', _FILE_NAME), root_node)

    # sources
    source_node = doc.createElement('source')
    create_child_node(doc, 'database', attrs.get('source.database', _DATABASE_NAME), source_node)
    create_child_node(doc, 'annotation', attrs.get('source.annotation', _ANNOTATION), source_node)
    create_child_node(doc, 'image', attrs.get('source.annotation', _IMAGE_SOURCE), source_node)
    root_node.appendChild(source_node)

    # owners
    owner_node = doc.createElement('owner')
    # create_child_node(doc, 'flickrid', 'NULL', owner_node)
    create_child_node(doc, 'name', _AUTHOR, owner_node)
    root_node.appendChild(owner_node)

    # size
    size_node = doc.createElement('size')
    create_child_node(doc, 'width', attrs.get('size.width', 'NULL').__str__(), size_node)
    create_child_node(doc, 'height', attrs.get('size.height', 'NULL').__str__(), size_node)
    create_child_node(doc, 'depth', attrs.get('size.depth', '3').__str__(), size_node)
    root_node.appendChild(size_node)

    # segmented
    create_child_node(doc, 'segmented', _SEGMENTED, root_node)
    object_node = create_object_node(doc, attrs=attrs)
    root_node.appendChild(object_node)
    return doc


def generate_voc2007format_xml(xml_file_name='./my.xml', folder='JPEGImages', filename='000001.jpg', size_width=300,
                           size_height=300, size_depth=3, bndbox=[0, 0, 0, 0]):
    """generate xml in voc2007"""
    assert len(bndbox) == 4

    attrs = {'folder': folder,
             'filename': filename,
             'size.width': size_width,
             'size.height': size_height,
             'size.depth': size_depth,
             'bndbox': bndbox}

    file_path, _ = os.path.split(xml_file_name)
    assert xml_file_name.endswith('.xml') and os.path.exists(file_path)

    temp_doc = create_annotation_node(attrs=attrs)
    with open(xml_file_name, 'w') as f:
        temp_doc.writexml(f, addindent=' ' * 4, newl='\n', encoding='utf-8')


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('--label_loc_type_info_path', required=True, dest='label_loc_type_info_path',
                        action='store', help='label_loc_type_info_path')
    parser.add_argument('--voc2007_Annotations_folder', required=True, dest='voc2007_Annotations_folder',
                        action='store', help='voc2007_Annotations_folder')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    assert os.path.exists(args.label_loc_type_info_path) and os.path.exists(args.voc2007_Annotations_folder)
    df = pd.read_csv(args.label_loc_type_info_path)

    folder = args.voc2007_Annotations_folder

    for idx, row in df.iterrows():
        xml_file_path = '{}/{}.xml'.format(folder, row['dataSet_id'])
        # now, exchange bndbox, x,y
        generate_voc2007format_xml(xml_file_name=xml_file_path,
                                   folder='JPEGImages',
                                   filename='{}.jpg'.format(row['dataSet_id']),
                                   size_width=row['length_x'],
                                   size_height=row['length_y'],
                                   size_depth=1,
                                   bndbox=[row['y.min'], row['x.min'], row['y.max'], row['x.max']])



