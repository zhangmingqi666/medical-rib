#coding=utf-8
import pandas as pd
import os
import argparse
import sys
import imageio
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def output_independent_rib_data(only_one_rib_df=None,
                                output_independent_rib_folder=None,
                                patient_id=None,
                                label=None,
                                expected_shape=None,
                                output_format='.jpg'):

    output_independent_rib_path = "{}/{}-{}{}".format(output_independent_rib_folder, patient_id, label, output_format)
    tmp_df = only_one_rib_df.groupby(['x', 'y']).agg({'z': 'count'})
    tmp_df.columns = ['z.count']
    tmp_df.reset_index(inplace=True)
    # tmp_df_max = tmp_df['z.count'].max()
    # tmp_df['z.count'] = tmp_df['z.count'].apply(lambda x: x * 255 / tmp_df_max).astype(np.uint8)

    # location x, y
    res_arr = None
    if output_format is '.jpg':
        res_arr = np.zeros(expected_shape)
        res_arr[(tmp_df['x'].values, tmp_df['y'].values)] = tmp_df['z.count'].values
    else:
        res_arr = np.zeros(expected_shape)
        res_arr[(tmp_df['x'].values, tmp_df['y'].values)] = tmp_df['z.count'].values
        # raise NotImplementedError
    res_arr_max = res_arr.max()
    res_arr = res_arr.astype(np.float64) / res_arr_max
    res_arr = res_arr * 255
    img = res_arr.astype(np.uint8)
    imageio.imwrite(output_independent_rib_path, img)


def split_ribs_to_independent_rib(data_df=None, output_independent_rib_folder='', patient_id='', output_format='.jpg'):
    label_uniques = data_df['c'].unique()
    offset_df = data_df.groupby('c').agg({'x': ['min', 'max'],
                                          'y': ['min', 'max'],
                                          'z': ['min', 'max']})
    offset_df.columns = ['{}.{}'.format(e[0], e[1]) for e in offset_df.columns.tolist()]
    for e in ['x', 'y', 'z']:
        offset_df['length_{}'.format(e)] = offset_df['{}.max'.format(e)] - offset_df['{}.min'.format(e)] + 1
        offset_df['offset_{}'.format(e)] = offset_df['{}.min'.format(e)]

    offset_df.reset_index(inplace=True)
    # offset_df.rename(columns={'index': 'c'}, inplace=True)
    # offset_df['dataSet_id'] = offset_df['c'].apply(lambda x: "{}_{}".format(patient_id, x))

    print("patient id = {} split into {} ribs_obtain.".format(patient_id, len(offset_df)))

    for index, row in offset_df.iterrows():

        label = row['c']
        only_one_rib_df = data_df[data_df['c'] == label]
        x_min, y_min, z_min = row['x.min'], row['y.min'], row['z.min']
        x_length, y_length, _ = row['length_x'], row['length_y'], row['length_z']
        for key, offset in [('x', x_min), ('y', y_min), ('z', z_min)]:
            only_one_rib_df[key] = only_one_rib_df[key].apply(lambda x: x-offset)

        output_independent_rib_data(only_one_rib_df=only_one_rib_df,
                                    output_independent_rib_folder=output_independent_rib_folder,
                                    patient_id=patient_id,
                                    label=label,
                                    expected_shape=(x_length, y_length),
                                    output_format=output_format)
    offset_df['dataSet_id'] = offset_df['c'].apply(lambda x: "{}-{}".format(patient_id, x))
    offset_df.drop(['x.min', 'y.min', 'z.min', 'x.max', 'y.max', 'z.max', 'c'], axis=1, inplace=True)

    return offset_df


def convert_all_ribs_to_independent_rib(in_folder='', output_independent_rib_folder='', output_format='.jpg'):
    files = os.listdir(in_folder)
    _data_set_offset_df = pd.DataFrame({})

    # print(output_independent_rib_folder)
    for file in files:
        if not file.endswith('.csv'):
            continue
        patient_id = file.replace('.csv', '')
        ribs_df = pd.read_csv(os.path.join(in_folder, file))

        if len(ribs_df) == 0:
            print("patient id = {}, ribs_obtain df 's length = 0".format(patient_id))
            continue

        offset_df = split_ribs_to_independent_rib(data_df=ribs_df,
                                                  output_independent_rib_folder=output_independent_rib_folder,
                                                  patient_id=patient_id,
                                                  output_format=output_format)

        _data_set_offset_df = _data_set_offset_df.append(offset_df)

    return _data_set_offset_df


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('--in_folder_path', required=True, dest='in_folder_path',
                        action='store', help='in_folder_path')
    parser.add_argument('--output_independent_rib_folder', required=True,
                        dest='output_independent_rib_folder', action='store',
                        help='output_independent_rib_folder')
    parser.add_argument('--dataset_offset_df_path', required=True,
                        dest='dataset_offset_df_path', action='store', help='dataset_offset_df_path')
    parser.add_argument('--output_format', required=True,
                        dest='output_format', action='store', help='output_format')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if not os.path.exists(args.in_folder_path) or not os.path.exists(args.output_independent_rib_folder):
        print('in folder or out folder dirs not exist')
        exit(1)

    data_set_offset_df = convert_all_ribs_to_independent_rib(in_folder=args.in_folder_path,
                                                             output_independent_rib_folder=args.output_independent_rib_folder,
                                                             output_format=args.output_format)
    data_set_offset_df.to_csv(args.dataset_offset_df_path, index=False)


