#coding=utf-8
import pandas as pd
import nibabel as nib
import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')


def read_excel(excel_path=None):
    """read (patient_id, location_id, rib_type) from **.xls"""
    df = pd.read_excel(excel_path)
    df = df[['id', 'location_id', 'type']]
    df = df.fillna(method='ffill')
    df['id'] = df['id'].astype(int).astype(str)
    # print(df)
    return df


def nii_read(nii_file_path=None):
    """read box min,max from nii file"""
    img = nib.load(nii_file_path)
    header = img.header
    pixel_zoom = header.get_zooms()
    img_arr = img.get_fdata()
    index = img_arr.nonzero()
    # exchange x,y
    tmp_df = pd.DataFrame({'y': index[0] / pixel_zoom[0],
                           'x': index[1] / pixel_zoom[1],
                           'z': index[2] / pixel_zoom[2]})
    x_min, x_max = int(tmp_df['x'].min()) + 1, int(tmp_df['x'].max()) + 1
    y_min, y_max = int(tmp_df['y'].min()) + 1, int(tmp_df['y'].max()) + 1
    z_min, z_max = int(tmp_df['z'].min()) + 1, int(tmp_df['z'].max()) + 1

    return {'x.max': x_max, 'x.min': x_min, 'y.max': y_max, 'y.min': y_min, 'z.max': z_max, 'z.min': z_min}


def location_read(folder_path=None):
    """read all the nii in folder_path"""
    location_df = pd.DataFrame(columns=('id', 'location_id', 'x.max', 'x.min', 'y.max', 'y.min', 'z.max', 'z.min'))
    for f in os.listdir(folder_path):
        next_dir = os.path.join(folder_path, f)
        if not os.path.isdir(next_dir):
            continue

        for file_name in os.listdir(next_dir):
            next_next_dir = os.path.join(next_dir, file_name)
            if not next_next_dir.endswith('.nii'):
                continue
            bounding_box = nii_read(nii_file_path=next_next_dir)
            new_row = {'id': f, 'location_id': file_name.replace('.nii', '')}
            new_row.update(bounding_box)
            location_df.loc[len(location_df)] = new_row
    return location_df


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Search some files')

    parser.add_argument('--excel_path',  dest='excel_path', action='store', help='excel_path', default=None)
    parser.add_argument('--nii_folder',  dest='nii_folder', action='store', help='excel_path', default=None)
    parser.add_argument('--output_path', dest='output_path', action='store', help='dcm_path', default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    location_df = None
    excel_df = None
    if os.path.exists(args.excel_path):
        excel_df = read_excel(excel_path=args.excel_path)

    if os.path.exists(args.nii_folder):
        location_df = location_read(folder_path=args.nii_folder)

    # if excel_df is not None and location_df is not None:
    assert (excel_df is not None and location_df is not None)

    label_df = location_df.merge(excel_df, on=['id', 'location_id'])
    label_df.to_csv(args.output_path, index=False)











