#coding=utf-8
import pandas as pd
import numpy as np
import os
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')


def read_excel(excel_path=None):
    """read (patient_id, location_id, rib_type) from **.xls"""
    df = pd.read_excel(excel_path, dtype={'id': np.str, 'location_id': np.str, 'type': np.str})
    df = df[['id', 'location_id', 'type']]
    df = df.fillna(method='ffill')
    return df


def one_ct_df_join_bounding_box(data_df=None, bounding_box_df=None, ct_id=''):
    dataset_map_label_df = pd.DataFrame(columns=('location_id', 'dataSet_id'))
    only_bounding_box_df = bounding_box_df[bounding_box_df['id'] == ct_id]
    for index, row in only_bounding_box_df.iterrows():
        box_x_min, box_x_max = row['box.x.min'], row['box.x.max']
        box_y_min, box_y_max = row['box.y.min'], row['box.y.max']
        box_z_min, box_z_max = row['box.z.min'], row['box.z.max']

        # print(x_min, x_max, y_min, y_max, z_min, z_max)
        temp_df = data_df[(data_df['x'] > box_x_min) & (data_df['x'] < box_x_max) & (data_df['y'] > box_y_min) &
                          (data_df['y'] < box_y_max) & (data_df['z'] > box_z_min) & (data_df['z'] < box_z_max)]
        mode = temp_df['c'].mode()
        if len(mode) == 1:
            dataset_map_label_df.loc[len(dataset_map_label_df)] = {'location_id': row['location_id'],
                                                                   'dataSet_id': "{}-{}".format(ct_id, mode[0])}
        else:
            # ribs_obtain are not all
            dataset_map_label_df.loc[len(dataset_map_label_df)] = {'location_id': row['location_id'],
                                                                   'dataSet_id': None}
    return dataset_map_label_df


def get_all_map_between_ct_and_location(csv_dataset_folder=None, bounding_box_df=None):
    """
    :param csv_dataset_folder:
    :param bounding_box_df:
    :return:
    """
    ct_id_arr = bounding_box_df['id'].unique()
    map_between_ct_and_location = pd.DataFrame({})
    for ct_id in ct_id_arr:

        if not os.path.exists("{}/{}.csv".format(csv_dataset_folder, ct_id)):
            continue

        data_df = pd.read_csv("{}/{}.csv".format(csv_dataset_folder, ct_id))
        # get global erea for every ribs.
        range_data_df = data_df.groupby('c').agg({'x': ['min', 'max'],
                                                  'y': ['min', 'max'],
                                                  'z': ['min', 'max']})
        range_data_df.columns = ['range.{}.{}'.format(e[0], e[1]) for e in range_data_df.columns.tolist()]
        for e in ['x', 'y', 'z']:
            range_data_df['range.{}.min'.format(e)] = range_data_df['range.{}.min'.format(e)].apply(lambda x: x-2)
            range_data_df['range.{}.max'.format(e)] = range_data_df['range.{}.max'.format(e)].apply(lambda x: x+2)
        range_data_df.reset_index(inplace=True)
        range_data_df.rename(columns={'index': 'dataSet_id', 'c': 'dataSet_id'}, inplace=True)
        range_data_df['dataSet_id'] = range_data_df['dataSet_id'].apply(lambda x: '{}-{}'.format(ct_id, x))

        if len(data_df) == 0:
            continue
        temp_map = one_ct_df_join_bounding_box(data_df=data_df, bounding_box_df=bounding_box_df, ct_id=ct_id)
        temp_map = temp_map.merge(range_data_df, on='dataSet_id', how='outer')
        map_between_ct_and_location = map_between_ct_and_location.append(temp_map)
    return map_between_ct_and_location


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('--ribs_df_cache_folder', dest='ribs_df_cache_folder', action='store',
                        help='ribs_df_cache_folder', default=None)
    parser.add_argument('--nii_loc_df_path', dest='nii_loc_df_path', action='store',
                        help='nii_loc_df_path', default=None)
    parser.add_argument('--rib_type_location_path', dest='rib_type_location_path', action='store',
                        help='rib_type_location_path', default=None)
    parser.add_argument('--data_join_label_path', dest='data_join_label_path', action='store',
                        help='data_join_label_path', default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    bounding_box_df = pd.read_csv(args.nii_loc_df_path)
    excel_df = read_excel(args.rib_type_location_path)
    map_df = get_all_map_between_ct_and_location(csv_dataset_folder=args.ribs_df_cache_folder, bounding_box_df=bounding_box_df)
    map_df = map_df.merge(excel_df, on='location_id', how='inner')
    map_df = map_df.merge(bounding_box_df, on='location_id', how='inner')
    map_df.to_csv(args.data_join_label_path, index=False)
