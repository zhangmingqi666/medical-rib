#coding=utf-8
import pandas as pd
import argparse
import sys
import os
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('--label_info_path', required=True, dest='label_info_path', action='store',
                        help='label_info_path')
    parser.add_argument('--dataset_offset_df_path', required=True, dest='dataset_offset_df_path', action='store',
                        help='dataset_offset_df_path')
    parser.add_argument('--data_join_label_path', required=True, dest='data_join_label_path', action='store',
                        help='data_join_label_path')
    parser.add_argument('--label_loc_type_info_path', required=True, dest='label_loc_type_info_path', action='store',
                        help='label_loc_type_info_path')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # assert os.path.exists(args.data_label_folder)
    # label_info_path = "{}/label_info.csv".format(args.data_label_folder)
    # dataset_offset_df_path = "{}/offset_df.csv".format(args.data_label_folder)
    # data_join_label_path = "{}/data_join_label.csv".format(args.data_label_folder)
    # label_loc_type_info_path = "{}/data_join_label.csv".format(args.data_label_folder)

    label_info_df = pd.read_csv(args.label_info_path)
    data_join_label_df = pd.read_csv(args.data_join_label_path)
    data_join_label_df = data_join_label_df.dropna(how='any', axis=0)
    dataset_offset_df = pd.read_csv(args.dataset_offset_df_path)

    df = label_info_df.merge(data_join_label_df.merge(dataset_offset_df, on='dataSet_id', how='inner'),
                             on='location_id', how='inner')

    for e in ['x', 'y', 'z']:
        df['{}.min'.format(e)] = df['{}.min'.format(e)] - df['offset_{}'.format(e)]
        df['{}.max'.format(e)] = df['{}.max'.format(e)] - df['offset_{}'.format(e)]

    df.to_csv(args.label_loc_type_info_path, index=False)




