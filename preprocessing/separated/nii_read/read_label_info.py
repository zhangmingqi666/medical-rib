#coding=utf-8
import pandas as pd
import nibabel as nib
import argparse
import os
import sys
import warnings


import sys, os


def add_python_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

from skimage.measure import label
import skimage


add_python_path(os.getcwd())


warnings.filterwarnings('ignore')


def multi_box_detect(img_arr=None):

    return


def nii_read(nii_file_path=None, keep_slicing=True, new_spacing=[1, 1, 1]):
    """read box min,max from nii file"""
    tmp_df = None
    try:
        img = nib.load(nii_file_path)
        header = img.header
        pixel_zoom = header.get_zooms()
        if keep_slicing:
            new_spacing = [pixel_zoom[2], pixel_zoom[2], pixel_zoom[2]]

        print('spacing', new_spacing)
        print('zoom', pixel_zoom)
        ratio_scale = [1.0 * e / f for e, f in zip(new_spacing, pixel_zoom)]
        print('ratio', ratio_scale)
        img_arr = img.get_fdata()
        img_arr_label = skimage.measure.label(img_arr, connectivity=2)

        index = img_arr.nonzero()
        # print(index)
        # exchange x,y
        tmp_df = pd.DataFrame({'y': index[0] * ratio_scale[0],
                               'x': index[1] * ratio_scale[1],
                               'z': index[2] * ratio_scale[2],
                               'box-c': img_arr[index]})
        #temp_df.to_csv("./data/nii_csv_files")
        # @tmp_df.to_csv(, index=False)
        # x_min, x_max = int(tmp_df['x'].min()) + 1, int(tmp_df['x'].max()) + 1
        # y_min, y_max = int(tmp_df['y'].min()) + 1, int(tmp_df['y'].max()) + 1
        # z_min, z_max = int(tmp_df['z'].min()) + 1, int(tmp_df['z'].max()) + 1
    except Exception as e:
        return None
    return tmp_df


def location_read(folder_path=None, keep_slicing=True):
    """read all the nii in folder_path"""
    location_df = pd.DataFrame(columns=('id', 'location_id', 'box.x.max', 'box.x.min',
                                        'box.y.max', 'box.y.min', 'box.z.max', 'box.z.min'))
    import re
    pattern = re.compile('^[a-zA-Z1-9].*nii')
    for f in os.listdir(folder_path):
        next_dir = os.path.join(folder_path, f)
        if not os.path.isdir(next_dir):
            continue

        print("read folder {}".format(f))
        for file_name in os.listdir(next_dir):

            next_next_dir = os.path.join(next_dir, file_name)
            if pattern.search(file_name) is None:
                continue

            if not file_name.__contains__('-'):
                continue

            print("read nii {}".format(file_name))

            temp_df = nii_read(nii_file_path=next_next_dir, keep_slicing=keep_slicing)

            if temp_df is not None:
                temp_df.to_csv("./data/nii_csv_files/{}.csv".format(file_name.replace('.nii', '')), index=False)
            box_df = temp_df.groupby('box-c').agg({'x': ['min', 'max'],
                                                   'y': ['min', 'max'],
                                                   'z': ['min', 'max']})
            box_df['location_id'] = file_name.replace('.nii', '')
            box_df['id'] = f
            location_df = location_df.append(box_df)

    return location_df


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Search some files')

    parser.add_argument('--nii_folder_list', nargs='+', dest='nii_folder_list', action='store')
    parser.add_argument('--output_path', dest='output_path', action='store', help='output_path', default=None)
    parser.add_argument('--keep_slicing', dest='keep_slicing', action='store', type=int,
                        help='keep_slicing', default=False)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    nii_folder_list = args.nii_folder_list
    location_df = pd.DataFrame({})
    keep_slicing = True if args.keep_slicing != 0 else False
    print("keep_slicing is {}".format(keep_slicing))
    for folder in nii_folder_list:
        temp_df = location_read(folder_path=folder, keep_slicing=keep_slicing)
        location_df = location_df.append(temp_df)

    location_df.to_csv(args.output_path, index=False, columns=['id', 'location_id', 'x.max',
                                                               'x.min', 'y.max', 'y.min', 'z.max', 'z.min'])













