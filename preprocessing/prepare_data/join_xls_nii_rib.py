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
    df = pd.read_excel(excel_path, dtype={'id': np.str, 'location_id': np.str, 'type': np.str, 'cnt':np.int},
                       na_values=['nan', 'NaN', np.NAN, np.nan])
    df = df[['id', 'location_id', 'type', 'cnt']]
    df['id'] = df['id'].replace('nan', np.NAN)
    df = df.fillna(method='ffill', axis=0)
    return df


def one_ct_df_join_one_bounding_box(data_df=None, _bounding_box_df=None, location_id=None):

    in_box_df = pd.DataFrame({})
    only_bounding_box_df = _bounding_box_df[bounding_box_df['location_id'] == location_id]
    for index, row in only_bounding_box_df.iterrows():
        box_x_min, box_x_max = row['box.x.min'], row['box.x.max']
        box_y_min, box_y_max = row['box.y.min'], row['box.y.max']
        box_z_min, box_z_max = row['box.z.min'], row['box.z.max']

        temp_df = data_df[(data_df['x'] > box_x_min) & (data_df['x'] <= box_x_max) & (data_df['y'] > box_y_min) &
                          (data_df['y'] <= box_y_max) & (data_df['z'] > box_z_min) & (data_df['z'] <= box_z_max)]
        in_box_df = in_box_df.append(temp_df)

    if len(in_box_df) == 0:
        print("Error:{} box can not cover ribs or ribs unavailable".format(location_id))
        return None

    hist_df = in_box_df['c'].value_counts().reset_index()
    c1, c1_count = hist_df.loc[0, ['index', 'c']]
    c2, c2_count = (0, 0) if len(hist_df) == 1 else hist_df.loc[1, ['index', 'c']]

    if c1_count < 2*c2_count:
        print("Error:{} ribs cannot dominate bounding box {}".format(c1, location_id))

    return c1


def get_all_map_between_ct_and_location(csv_dataset_folder=None, bounding_box_df=None):
    """
    :param csv_dataset_folder:
    :param bounding_box_df:
    :return:
    """
    ct_id_arr = bounding_box_df['id'].unique()
    map_ct_id_list = []
    map_location_id_list = []
    map_data_id_list = []

    for ct_id in ct_id_arr:

        ct_data_df_path = "{}/{}.csv".format(csv_dataset_folder, ct_id)
        if not os.path.exists(ct_data_df_path):
            print("error: ct data {} not exist.".format(ct_id))
            continue

        data_df = pd.read_csv(ct_data_df_path, dtype={'x': np.int, 'y': np.int, 'z': np.int, 'c': np.str})

        # warning
        if len(data_df) < 10000:
            print("error: ct data {} very few.".format(ct_id))
            continue

        """
        # get ribs local area.
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
        """

        all_box_for_ct_id_df = bounding_box_df[bounding_box_df['id'] == ct_id]
        for location_id in all_box_for_ct_id_df['location_id'].unique():
            rib_id = one_ct_df_join_one_bounding_box(data_df=data_df, _bounding_box_df=bounding_box_df,
                                                     location_id=location_id)
            if rib_id is None:
                continue

            map_ct_id_list.append(ct_id)
            map_location_id_list.append(location_id)
            map_data_id_list.append('{}-{}'.format(ct_id, rib_id))

    return pd.DataFrame({'id': map_ct_id_list, 'location_id': map_location_id_list, 'dataSet_id': map_data_id_list})


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

    """
        excel_df = read_excel("/Users/jiangyy/projects/medical-rib/data/csv_files/rib_type_location.xls")
    
        print(len(excel_df))
        location_df = pd.read_csv("/Users/jiangyy/projects/medical-rib/data/csv_files/nii_loc_df.csv")
        location_df_cnt = location_df.groupby(['id', 'location_id']).agg({'location_id': ['count']})
        location_df_cnt.columns = ['box.count']
        location_df_cnt.reset_index(inplace=True)
    """
    print('Called with args:')
    print(args)
    bounding_box_df = pd.read_csv(args.nii_loc_df_path, dtype={'id': np.str, 'location_id': np.str,
                                                               'box.x.max': np.int, 'box.x.min': np.int,
                                                               'box.y.max': np.int, 'box.y.min': np.int,
                                                               'box.z.max': np.int, 'box.z.min': np.int})

    # bounding_box_df_hist = bounding_box_df.groupby(['id', 'location_id']).agg({'location_id': ['count']})
    # bounding_box_df_hist.columns = ['box.count']
    # bounding_box_df_hist.reset_index(inplace=True)

    excel_label_df = read_excel(args.rib_type_location_path)
    map_df = get_all_map_between_ct_and_location(csv_dataset_folder=args.ribs_df_cache_folder, bounding_box_df=bounding_box_df)
    map_df.to_csv(args.data_join_label_path, index=False)

    # logs

