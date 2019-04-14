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
    tmp_df = only_one_rib_df.groupby(['z', 'y']).agg({'z': 'count'})
    tmp_df.columns = ['z.count']
    tmp_df.reset_index(inplace=True)
    # tmp_df_max = tmp_df['z.count'].max()
    # tmp_df['z.count'] = tmp_df['z.count'].apply(lambda x: x * 255 / tmp_df_max).astype(np.uint8)

    # location x, y
    #res_arr = None
    #if output_format is '.jpg':
    res_arr = np.zeros(expected_shape)
    res_arr[(tmp_df['z'].values, tmp_df['y'].values)] = tmp_df['z.count'].values
    #else:
    #    res_arr = np.zeros(expected_shape)
    #    res_arr[(tmp_df['x'].values, tmp_df['y'].values)] = tmp_df['z.count'].values
        # raise NotImplementedError
    res_arr_max = res_arr.max()
    res_arr = res_arr.astype(np.float64) / res_arr_max
    res_arr = res_arr * 255
    img = res_arr.astype(np.uint8)
    imageio.imwrite(output_independent_rib_path, img)


def split_ribs_to_independent_rib(data_df=None, output_independent_rib_folder='', patient_id='', output_format='.jpg'):

    # get the range min/max for every ribs
    range_data_df = data_df.groupby('c').agg({'x': ['min', 'max'],
                                              'y': ['min', 'max'],
                                              'z': ['min', 'max']})
    range_data_df.columns = ['range.{}.{}'.format(e[0], e[1]) for e in range_data_df.columns.tolist()]

    def f(row, e):
        return row['range.{}.max'.format(e)] - row['range.{}.min'.format(e)]
    for e in ['x', 'y', 'z']:
        range_data_df['range.{}.min'.format(e)] = range_data_df['range.{}.min'.format(e)].apply(lambda x: x - 2)
        range_data_df['range.{}.max'.format(e)] = range_data_df['range.{}.max'.format(e)].apply(lambda x: x + 2)
        range_data_df['range.{}.length'.format(e)] = range_data_df.apply(lambda row: f(row, e), axis=1)
    range_data_df.reset_index(inplace=True)
    range_data_df.rename(columns={'index': 'c'})

    print("patient id = {} split into {} ribs_obtain.".format(patient_id, len(range_data_df)))

    local_data_df = data_df.merge(range_data_df, on='c')
    for e in ['x', 'y', 'z']:
        local_data_df[e] = local_data_df.apply(lambda row: row[e]-row['range.{}.min'.format(e)], axis=1).apply(np.int64)
    for index, line in range_data_df.iterrows():

        class_id = line['c']
        output_independent_rib_data(only_one_rib_df=local_data_df[local_data_df['c'] == class_id],
                                    output_independent_rib_folder=output_independent_rib_folder,
                                    patient_id=patient_id,
                                    label=class_id,
                                    expected_shape=(line['range.z.length'], line['range.y.length']),
                                    output_format=output_format)


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

        split_ribs_to_independent_rib(data_df=ribs_df,
                                      output_independent_rib_folder=output_independent_rib_folder,
                                      patient_id=patient_id,
                                      output_format=output_format)


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

    convert_all_ribs_to_independent_rib(in_folder=args.in_folder_path,
                                        output_independent_rib_folder=args.output_independent_rib_folder,
                                        output_format=args.output_format)


