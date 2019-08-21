#coding=utf-8
import pandas as pd
import argparse
import imageio
import numpy as np
import warnings
import sys, os

warnings.filterwarnings('ignore')


def read_excel(excel_path=None):
    """read (patient_id, location_id, rib_type) from **.xls"""
    df = pd.read_excel(excel_path, dtype={'id': np.str, 'location_id': np.str, 'type': np.str, 'cnt':np.int},
                       na_values=['nan', 'NaN', np.NAN, np.nan])
    df = df[['id', 'location_id', 'type', 'cnt']]
    df['id'] = df['id'].replace('nan', np.NAN)
    df = df.fillna(method='ffill', axis=0)
    return df


def get_rib_data_depend_rib_id(csv_dataset_folder=None, rib_id=None, ct_id=None):

    ct_data_df_path = "{}/{}.csv".format(csv_dataset_folder, ct_id)
    if not os.path.exists(ct_data_df_path):
        print("error: ct data {} not exist.".format(ct_id))
        return None

    data_df = pd.read_csv(ct_data_df_path, dtype={'x': np.int, 'y': np.int, 'z': np.int, 'c': np.str})
    _rib_data_df = data_df[data_df['c'] == rib_id[(len(ct_id)+1):]]
    return _rib_data_df


# def get_rib_data_area(rib_data_df=None):
#     range_dict = {}
#     for e in ['x', 'y', 'z']:
#         range_dict['range.{}.min'.format(e)] = rib_data_df[e].min() - 2
#         range_dict['range.{}.max'.format(e)] = rib_data_df[e].max() + 2
#     return range_dict
#
#
# def transfer_glb_rib_2_local_rib(rib_data_df=None, range_dict={}):
#     for e in ['x', 'y', 'z']:
#         rib_data_df[e] = rib_data_df[e].apply(lambda x: x - range_dict['range.%s.min' % e]).apply(np.int64)
#     return rib_data_df
#
#
# def generate_filename_4_xml_jpg(ct_id, rib_id, location_list):
#     return rib_id


def generate_filename_4_xml_jpg_2(ct_id, rib_id, location_list):
    location_part = "_".join([x[(len(ct_id)+1):] for x in location_list])
    rib_part = rib_id[(len(ct_id)+1):]
    #return '-'.join([ct_id, rib_part, location_part])
    return '-'.join([ct_id, location_part])


# def mk_boxes_tighten(_locations_for_ribs=None, _rib_data_df=None):
#
#     new_locations_for_ribs = pd.DataFrame(columns=["id", "location_id", "box.x.max", "box.x.min", "box.y.max",
#                                                    "box.y.min", "box.z.max", "box.z.min"])
#     for _, _row in _locations_for_ribs.iterrows():
#
#         box_x_min, box_x_max = _row['box.x.min'], _row['box.x.max']
#         box_y_min, box_y_max = _row['box.y.min'], _row['box.y.max']
#         box_z_min, box_z_max = _row['box.z.min'], _row['box.z.max']
#
#         temp_df = _rib_data_df[(_rib_data_df['x'] >= box_x_min) & (_rib_data_df['x'] <= box_x_max) &
#                                (_rib_data_df['y'] >= box_y_min) & (_rib_data_df['y'] <= box_y_max) &
#                                (_rib_data_df['z'] >= box_z_min) & (_rib_data_df['z'] <= box_z_max)]
#         if len(temp_df) < 10:
#             print("A box in {} has low data, so dropped".format(_row['location_id']))
#             continue
#
#         new_locations_for_ribs.loc[len(new_locations_for_ribs)] = {'id': _row['id'], 'location_id': _row['id'],
#                                                                    'box.x.max': min(box_x_max, temp_df['x'].max()),
#                                                                    'box.x.min': max(box_x_min, temp_df['x'].min()),
#                                                                    'box.y.max': min(box_y_max, temp_df['y'].max()),
#                                                                    'box.y.min': max(box_y_min, temp_df['y'].min()),
#                                                                    'box.z.max': min(box_z_max, temp_df['z'].max()),
#                                                                    'box.z.min': max(box_z_min, temp_df['z'].min())}
#     return new_locations_for_ribs

#
# def merge_connected_boxes(boxes=[]):
#     return

#
# def parse_args():
#     """
#     Parse input arguments
#     """
#     parser = argparse.ArgumentParser(description='Search some files')
#
#     parser.add_argument('--ribs_df_cache_folder', dest='ribs_df_cache_folder', action='store',
#                         help='ribs_df_cache_folder', default=None)
#     parser.add_argument('--nii_loc_df_path', dest='nii_loc_df_path', action='store',
#                         help='nii_loc_df_path', default=None)
#     parser.add_argument('--rib_type_location_path', dest='rib_type_location_path', action='store',
#                         help='rib_type_location_path', default=None)
#     parser.add_argument('--data_join_label_path', dest='data_join_label_path', action='store',
#                         help='data_join_label_path', default=None)
#     parser.add_argument('--voc2007_Annotations_folder', required=True, dest='voc2007_Annotations_folder',
#                         action='store', help='voc2007_Annotations_folder')
#     parser.add_argument('--output_independent_rib_folder', required=True,
#                         dest='output_independent_rib_folder', action='store',
#                         help='output_independent_rib_folder')
#     parser.add_argument('--project_method', required=True,
#                         dest='project_method', action='store',
#                         help='project_method')
#     parser.add_argument('--output_format', required=True,
#                         dest='output_format', action='store', help='output_format')
#
#     if len(sys.argv) == 1:
#         parser.print_help()
#         sys.exit(1)
#
#     return parser.parse_args()


if __name__ == '__main__':
    # args = parse_args()
    #
    # print('Called with args:')
    # print(args)
    #
    #map_df = pd.read_csv(args.data_join_label_path)
    import os
    M_path = '/home/jiangyy/projects/medical-rib/data'
    map_df = pd.read_csv(os.path.join(M_path, 'csv_files', 'join_label.csv'))
    map_unique_df = map_df.groupby(['id', 'dataSet_id']).agg({'id': 'count'}).\
        rename(columns={'id': 'count'}).reset_index()

    bounding_box_df = pd.read_csv(os.path.join(M_path, 'csv_files', 'nii_loc_df.csv'),
                                  dtype={'id': np.str, 'location_id': np.str,
                                         'box.x.max': np.int, 'box.x.min': np.int,
                                         'box.y.max': np.int, 'box.y.min': np.int,
                                         'box.z.max': np.int, 'box.z.min': np.int})

    for _, row in map_unique_df.iterrows():
        ct_id, rib_id = row['id'], row['dataSet_id']
        map_loc_df = map_df[map_df['dataSet_id'] == rib_id]

        if len(map_loc_df) > 1:
            print("error: {} rib data boxes from {} labels".format(rib_id, len(map_loc_df)))
            continue

        locations_unique = map_loc_df['location_id'].unique()

        rib_data_df_path = os.path.join(M_path, 'ribs_df_cache', '{}.csv'.format(ct_id))
        if not os.path.exists(rib_data_df_path):
            print("error: ct data {} not exist.".format(ct_id))
            continue

        data_df = pd.read_csv(rib_data_df_path, dtype={'x': np.int, 'y': np.int, 'z': np.int,
                                                       'c': np.str, 'v': np.float32})
        rib_data_df = data_df[data_df['c'] == rib_id[(len(ct_id) + 1):]]

        kitti_depth_name = os.path.join(M_path, 'kitti', 'depth', '{}.txt'.format(rib_id))
        np.savetxt(kitti_depth_name, rib_data_df[['x', 'y', 'z', 'v']].values, delimiter=' ', fmt='%.3f')

        locations_for_ribs = bounding_box_df[bounding_box_df['location_id'].isin(locations_unique)]
        if len(locations_for_ribs) > len(map_unique_df):
            print("In {}, some locations has more than 1 boxes".format(rib_id))

        locations_for_ribs['type'] = 'rib_hurt'
        kitti_label_name = os.path.join(M_path, 'kitti', 'label', '{}.txt'.format(rib_id))
        locations_for_ribs.to_csv(kitti_label_name, columns=['type', 'box.x.min', 'box.x.max', 'box.y.min',
                                                             'box.y.max', 'box.z.min', 'box.z.max'],
                                  header=False, index=False, sep=' ')

