from enum import Enum
import matplotlib.pyplot as plt
import gc
import numpy as np
from interval import Interval
import warnings
import sys, os
import pandas as pd


def add_python_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_python_path(os.getcwd())
# from projects
warnings.filterwarnings('ignore')


class BonePredict:

    def __init__(self, bone_data=None, arr_shape=None, spine_width=100, rib_diameter=30,
                 spine_connected_rib_y_length_thresholds=180, through_thresholds=0.6,
                 prior_zoy_center_y_axis_line_df=None, detection_objective='rib'):
        self.bone_data = bone_data
        self.arr_shape = arr_shape
        self.location_info_df = None
        self.calc_features_for_all_bones()
        pass

    def calc_features_for_all_bones(self):

        # calc features
        location_info_df = self.bone_data.groupby('c').agg({'x': ['min', 'max', 'mean'],
                                                            'y': ['min', 'max', 'mean'],
                                                            'z': ['min', 'max', 'mean'],
                                                            'c': 'count'})
        location_info_df.columns = ['%s.%s' % e for e in location_info_df.columns.tolist()]

        """
            mean_z_distance_on_xoy,
            std_z_distance_on_xoy,
            std_z_distance_div_mean_z_distance,
        """
        info_df_on_xoy = self.bone_data.groupby(['c', 'x', 'y']).agg({'z': ['min', 'max']})
        info_df_on_xoy.columns = ['%s.%s' % e for e in info_df_on_xoy.columns.tolist()]
        info_df_on_xoy['length'] = info_df_on_xoy.apply(lambda row: row['z.max'] - row['z.min'], axis=1)
        info_df_on_xoy.reset_index(inplace=True)
        info_df_on_xoy_grp = info_df_on_xoy.groupby('c').agg({'length': ['count', 'std', 'mean']})
        info_df_on_xoy_grp.columns = ['%s.%s' % e for e in info_df_on_xoy_grp.columns.tolist()]
        info_df_on_xoy_grp.rename(columns={'length.count': 'length.count.on.xoy',
                                           'length.std': 'std_z_distance_on_xoy',
                                           'length.mean': 'mean_z_distance_on_xoy'}, inplace=True)

        def f1(row):
            return row['std_z_distance_on_xoy'] / row['mean_z_distance_on_xoy']
        info_df_on_xoy_grp['std_z_distance_div_mean_z_distance'] = info_df_on_xoy_grp.apply(lambda row: f1(row), axis=1)
        location_info_df = pd.concat([location_info_df, info_df_on_xoy_grp], axis=1)

        """
            mean_x_distance_on_zoy,
            std_x_distance_on_zoy,
            std_x_distance_div_mean_x_distance,
        """
        info_df_on_zoy = self.bone_data.groupby(['c', 'z', 'y']).agg({'x': ['min', 'max']})
        info_df_on_zoy.columns = ['%s.%s' % e for e in info_df_on_zoy.columns.tolist()]
        info_df_on_zoy['length'] = info_df_on_zoy.apply(lambda row: row['x.max'] - row['x.min'], axis=1)
        info_df_on_zoy.reset_index(inplace=True)
        info_df_on_zoy_grp = info_df_on_zoy.groupby('c').agg({'length': ['std', 'mean']})
        info_df_on_zoy_grp.columns = ['%s.%s' % e for e in info_df_on_zoy_grp.columns.tolist()]
        info_df_on_zoy_grp.rename(columns={'length.std': 'std_x_distance_on_zoy',
                                           'length.mean': 'mean_x_distance_on_zoy'}, inplace=True)

        def f2(row):
            return row['std_x_distance_on_zoy'] / row['mean_x_distance_on_zoy']
        info_df_on_zoy_grp['std_x_distance_div_mean_x_distance'] = info_df_on_zoy_grp.apply(lambda row: f2(row), axis=1)
        location_info_df = pd.concat([location_info_df, info_df_on_zoy_grp], axis=1)

        """
            mean_y_distance_on_zox,
            std_y_distance_on_zox,
            std_y_distance_div_mean_y_distance,       
        """
        info_df_on_zox = self.bone_data.groupby(['c', 'z', 'x']).agg({'y': ['min', 'max']})
        info_df_on_zox.columns = ['%s.%s' % e for e in info_df_on_zox.columns.tolist()]
        info_df_on_zox['length'] = info_df_on_zox.apply(lambda row: row['y.max'] - row['y.min'], axis=1)
        info_df_on_zox.reset_index(inplace=True)
        info_df_on_zox_grp = info_df_on_zox.groupby('c').agg({'length': ['std', 'mean']})
        info_df_on_zox_grp.columns = ['%s.%s' % e for e in info_df_on_zox_grp.columns.tolist()]
        info_df_on_zox_grp.rename(columns={'length.std': 'std_y_distance_on_zox',
                                           'length.mean': 'mean_y_distance_on_zox'}, inplace=True)

        def f3(row):
            return row['std_y_distance_on_zox'] / row['mean_y_distance_on_zox']
        info_df_on_zox_grp['std_y_distance_div_mean_y_distance'] = info_df_on_zox_grp.apply(lambda row: f3(row), axis=1)
        location_info_df = pd.concat([location_info_df, info_df_on_zox_grp], axis=1)

        # feature iou_on_xoy
        def f_iou(row):
            return 1.0 * row['length.count.on.xoy'] / ((row['x.max'] - row['x.min'])*(row['y.max'] - row['y.min']))
        location_info_df['iou_on_xoy'] = location_info_df.apply(lambda row: f_iou(row), axis=1)
        location_info_df.drop(columns=['length.count.on.xoy'], axis=1)

        distance_nearest_centroid_df = self.bone_data.merge(location_info_df, on='c')

        # feature distance_nearest_centroid_df_min
        def f_dis(row):
            return (row['x']-row['x.mean'])**2 + (row['y']-row['y.mean'])**2 + (row['z']-row['z.mean'])**2
        distance_nearest_centroid_df['distance_centroid'] = distance_nearest_centroid_df.apply(lambda row: f_dis(row), axis=1)
        distance_nearest_centroid_df_min = distance_nearest_centroid_df.groupby('c').agg({'distance_centroid': ['min']})
        distance_nearest_centroid_df_min.columns = ['distance_nearest_centroid']
        location_info_df = pd.concat([location_info_df, distance_nearest_centroid_df_min], axis=1)

        for shape, e in zip(self.arr_shape, ['z', 'x', 'y']):
            location_info_df['{}.min'.format(e)] = location_info_df['{}.min'.format(e)].apply(lambda x: x / shape)
            location_info_df['{}.max'.format(e)] = location_info_df['{}.max'.format(e)].apply(lambda x: x / shape)
            location_info_df['{}.mean'.format(e)] = location_info_df['{}.mean'.format(e)].apply(lambda x: x / shape)
            location_info_df['{}.length'.format(e)] = location_info_df.apply(lambda row: row['{}.max'.format(e)] -
                                                                             row['{}.min'.format(e)], axis=1)

        location_info_df.reset_index(inplace=True)
        location_info_df.rename(columns={'x.min': 'x_min/x_shape', 'y.min': 'y_min/y_shape', 'z.min': 'z_min/z_shape',
                                         'x.max': 'x_max/x_shape', 'y.max': 'y_max/y_shape', 'z.max': 'z_max/z_shape',
                                         'x.mean': 'x_centroid/x_shape', 'y.mean': 'y_centroid/y_shape',
                                         'z.mean': 'z_centroid/z_shape',
                                         'x.length': 'x_length/x_shape', 'y.length': 'y_length/y_shape',
                                         'z.length': 'z_length/z_shape',
                                         'c.count': 'point_count',
                                         'index': 'class_id', 'c': 'class_id'},
                                inplace=True)
        self.location_info_df = location_info_df

    def get_features_for_all_bones(self):
        return self.location_info_df

    def plot_bone(self, class_id=None, save=False, save_path=None):
        img_yzd_arr = np.zeros((self.arr_shape[0], self.arr_shape[1]))
        img_2d_df = self.bone_data[self.bone_data['c'] == class_id].groupby(['y', 'z']).agg({'x': ['count']})
        img_2d_df.columns = ['x.count']
        img_2d_df.reset_index(inplace=True)
        img_yzd_arr[(img_2d_df['z'].values, img_2d_df['y'].values)] = img_2d_df['x.count'].values
        plt.imshow(img_yzd_arr)
        if save:
            plt.savefig(save_path)
        else:
            plt.show()

        del img_2d_df, img_yzd_arr
        gc.collect()
        

    """
    def print_bone_info(self):
        print('#'*30 + ' bone ' + '#' * 30)
        print('bone direction:', self.left_or_right())
        print('bone Type:', self.bone_type)
        print('rib type:', self.rib_type)
        print('iou = %.2f' % self.get_iou_on_xoy())
        print('z_distance_max is', self.get_features_z_distance_on_xoy())
        print('nearest point is', self.get_distance_between_centroid_and_nearest_point())
        print('through center line rate = {}'.format(self.center_line_through_spine_prob))
    """
