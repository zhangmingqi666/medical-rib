#coding=utf-8
import pandas as pd
import os
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')


def one_ct_df_join_bounding_box(data_df=None, bounding_box_df=None, ct_id=''):
    dataset_map_label_df = pd.DataFrame(columns=('location_id', 'dataSet_id'))
    only_bounding_box_df = bounding_box_df[bounding_box_df['id'] == ct_id]
    for index, row in only_bounding_box_df.iterrows():
        # exchange x with y, label reason.
        x_min, x_max = row['x.min'], row['x.max']
        y_min, y_max = row['y.min'], row['y.max']
        z_min, z_max = row['z.min'], row['z.max']
        # print(x_min, x_max, y_min, y_max, z_min, z_max)
        temp_df = data_df[(data_df['x'] > x_min) & (data_df['x'] < x_max) & (data_df['y'] > y_min) &
                          (data_df['y'] < y_max) & (data_df['z'] > z_min) & (data_df['z'] < z_max)]
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
        if len(data_df) == 0:
            continue
        temp_map = one_ct_df_join_bounding_box(data_df=data_df, bounding_box_df=bounding_box_df, ct_id=ct_id)
        map_between_ct_and_location = map_between_ct_and_location.append(temp_map)
    return map_between_ct_and_location


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('--rib_only_cache_folder', dest='rib_only_cache_folder', action='store',
                        help='rib_only_cache_folder', default=None)
    parser.add_argument('--label_info_path', dest='label_info_path', action='store', help='label_info_path', default=None)
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

    if os.path.exists(args.rib_only_cache_folder) and os.path.exists(args.label_info_path) and args.data_join_label_path is not None:
        bounding_box_df = pd.read_csv(args.label_info_path)
        map_df = get_all_map_between_ct_and_location(csv_dataset_folder=args.rib_only_cache_folder, bounding_box_df=bounding_box_df)
        print(map_df)
        map_df.to_csv(args.data_join_label_path, index=False)