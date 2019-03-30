from enum import Enum
import matplotlib.pyplot as plt
import gc
import numpy as np
import pandas as pd
import warnings
import sys, os
import skimage
from skimage.measure import label


def add_python_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_python_path(os.getcwd())
# from projects
from preprocessing.separated.ribs_obtain.util import (sparse_df_to_arr, arr_to_sparse_df, sparse_df_remove_min)
warnings.filterwarnings('ignore')


class BonePredict:

    def __init__(self, bone_data=None, arr_shape=None, spine_width=100, rib_diameter=30,
                 spine_connected_rib_y_length_thresholds=180, through_thresholds=0.6,
                 prior_zoy_center_y_axis_line_df=None, detection_objective='rib'):
        # DataFrame
        self.bone_data = bone_data
        self.arr_shape = arr_shape

        self.z_mid_line, self.half_z_shape = arr_shape[0] / 2, arr_shape[0] / 2
        self.x_mid_line, self.half_x_shape = arr_shape[1] / 2, arr_shape[1] / 2
        self.y_mid_line, self.half_y_shape = arr_shape[2] / 2, arr_shape[2] / 2

        # basic feature('min', 'max', 'length', 'centroid') about x,y,z axis
        self.basic_axis_feature = None
        self.set_basic_axis_feature()

        # calc y Max(Max-Min),Mean(Max-Min) group by y, judge whether connected with ribs_obtain
        # self.y_length_statistics_on_z = None
        # self.set_y_length_statistics_on_z()

        # IOU calculation
        self.bone_iou_on_xoy = None
        self.calc_iou_on_xoy()

        # calc distance between centroid and nearest point.
        self.distance_nearest_centroid = None

        # z length distribution for bone
        self.mean_z_distance_on_xoy = None  # calc Z Mean(Max-Min) group by (x,y)
        self.std_z_distance_on_xoy = None  # calc Z std(Max-Min) group by (x,y)
        self.z_distance_std_div_z_distance_mean = None

        # x length distribution for bone
        self.mean_x_distance_on_zoy = None  # calc X Mean(Max-Min) group by (z,y)
        self.std_x_distance_on_zoy = None  # calc X std(Max-Min) group by (z,y)
        self.x_distance_std_div_x_distance_mean = None

        # y length distribution for bone
        self.mean_y_distance_on_zox = None  # calc Y Mean(Max-Min) group by (z,x)
        self.std_y_distance_on_zox = None  # calc Y std(Max-Min) group by (z,x)
        self.y_distance_std_div_y_distance_mean = None

        # calc distance between centroid and nearest point.
        self.calc_distance_between_centroid_and_nearest_point()

        # calc Z Max(Max-Min) group by (x,y)
        self.set_features_z_distance_on_xoy()
        self.set_features_x_distance_on_zoy()
        self.set_features_y_distance_on_zox()

        # self.multi_ribs = False
        # self.detect_multi_ribs()

    def get_bone_data(self):
        return self.bone_data

    def set_basic_axis_feature(self):
        bone_data_min = self.bone_data['z'].min(), self.bone_data['x'].min(), self.bone_data['y'].min()
        bone_data_max = self.bone_data['z'].max(), self.bone_data['x'].max(), self.bone_data['y'].max()
        bone_data_length = tuple(np.array(bone_data_max) - np.array(bone_data_min) + 1)
        centroid = self.bone_data['z'].mean(), self.bone_data['x'].mean(), self.bone_data['y'].mean()
        self.basic_axis_feature = {'min': bone_data_min,
                                   'max': bone_data_max,
                                   'length': bone_data_length,
                                   'centroid': centroid}

    def get_basic_axis_feature(self, feature='centroid'):
        assert feature in self.basic_axis_feature
        return self.basic_axis_feature.get(feature)

    """
    used for calc whether some ribs_obtain connected with spine.
    """
    def set_y_length_statistics_on_z(self):

        y_distance = self.bone_data.groupby('z').agg({'y': ['max', 'min']})
        y_distance.columns = ['%s.%s' % e for e in y_distance.columns.tolist()]
        y_distance['y_distance'] = y_distance['y.max'] - y_distance['y.min']
        self.y_length_statistics_on_z = {'max': y_distance['y_distance'].max(),
                                         'mean': y_distance['y_distance'].mean(),
                                         'std': y_distance['y_distance'].std()}
        del y_distance

    def get_y_length_statistics_on_z(self, feature='min'):
        assert feature in self.y_length_statistics_on_z
        return self.y_length_statistics_on_z.get(feature)

    def calc_distance_between_centroid_and_nearest_point(self):
        z_centroid, x_centroid, y_centroid = self.get_basic_axis_feature(feature='centroid')

        def f(row):
            return np.sqrt((row['z'] - z_centroid)**2 + (row['x'] - x_centroid)**2 + (row['y'] - y_centroid)**2)
        # warning
        self.bone_data['dis_nearest'] = self.bone_data.apply(lambda row: f(row), axis=1)
        self.distance_nearest_centroid = self.bone_data['dis_nearest'].min()

    def get_distance_between_centroid_and_nearest_point(self):
        return self.distance_nearest_centroid

    def set_features_z_distance_on_xoy(self):
        z_distance = self.bone_data.groupby(['x', 'y']).agg({'z': ['max', 'min']})
        z_distance.columns = ['%s.%s' % e for e in z_distance.columns.tolist()]
        z_distance['z_distance'] = z_distance['z.max'] - z_distance['z.min']
        self.mean_z_distance_on_xoy = z_distance['z_distance'].mean()
        self.std_z_distance_on_xoy = z_distance['z_distance'].std()
        self.z_distance_std_div_z_distance_mean = (self.std_z_distance_on_xoy + 0.001) / (self.mean_z_distance_on_xoy + 0.001)
        del z_distance

    def set_features_x_distance_on_zoy(self):
        x_distance = self.bone_data.groupby(['z', 'y']).agg({'x': ['max', 'min']})
        x_distance.columns = ['%s.%s' % e for e in x_distance.columns.tolist()]
        x_distance['x_distance'] = x_distance['x.max'] - x_distance['x.min']
        self.mean_x_distance_on_zoy = x_distance['x_distance'].mean()
        self.std_x_distance_on_zoy = x_distance['x_distance'].std()
        self.x_distance_std_div_x_distance_mean = (self.std_x_distance_on_zoy + 0.001) / (self.mean_x_distance_on_zoy + 0.001)
        del x_distance

    def set_features_y_distance_on_zox(self):
        y_distance = self.bone_data.groupby(['z', 'x']).agg({'y': ['max', 'min']})
        y_distance.columns = ['%s.%s' % e for e in y_distance.columns.tolist()]
        y_distance['y_distance'] = y_distance['y.max'] - y_distance['y.min']
        self.mean_y_distance_on_zox = y_distance['y_distance'].mean()
        self.std_y_distance_on_zox = y_distance['y_distance'].std()
        self.y_distance_std_div_y_distance_mean = (self.std_y_distance_on_zox + 0.001) / (self.mean_y_distance_on_zox + 0.001)
        del y_distance

    def get_features_z_distance_on_xoy(self):
        features_z_distance_on_xoy = {'mean': self.mean_z_distance_on_xoy,
                                      'std': self.std_z_distance_on_xoy,
                                      'std_div_mean': self.z_distance_std_div_z_distance_mean
                                      }
        return features_z_distance_on_xoy

    def get_features_x_distance_on_zoy(self):
        features_x_distance_on_zoy = {'mean': self.mean_x_distance_on_zoy,
                                      'std': self.std_x_distance_on_zoy,
                                      'std_div_mean': self.x_distance_std_div_x_distance_mean
                                      }
        return features_x_distance_on_zoy

    def get_features_y_distance_on_zox(self):
        features_y_distance_on_zox = {'mean': self.mean_y_distance_on_zox,
                                      'std': self.std_y_distance_on_zox,
                                      'std_div_mean': self.y_distance_std_div_y_distance_mean
                                      }
        return features_y_distance_on_zox

    def calc_iou_on_xoy(self):
        pipeline_area = len(self.bone_data.groupby(['x', 'y']))
        _, x_length, y_length = self.get_basic_axis_feature(feature='length')
        rectangle_area = x_length * y_length
        self.bone_iou_on_xoy = pipeline_area / rectangle_area

    def get_iou_on_xoy(self):
        return self.bone_iou_on_xoy

    def cut_multi_ribs(self):
        _, x_min, _ = self.get_basic_axis_feature(feature='min')
        _, x_max, _ = self.get_basic_axis_feature(feature='max')

        map2d_df = self.bone_data.groupby(['x', 'z']).agg({'y': 'sum'})
        map2d_df.reset_index(inplace=True)

        map2d_image = np.ones((self.arr_shape[0], self.arr_shape[1]))
        map2d_image[(map2d_df['z'].values, map2d_df['x'].values)] = 0
        map2d_image[:, :((x_min + x_max)//2)] = 0
        label_arr = skimage.measure.label(map2d_image, connectivity=2)
        index = label_arr.nonzero()
        sparse_df = pd.DataFrame({'x': index[1],
                                  'z': index[0],
                                  'c': label_arr[index]})
        cluster_df = sparse_df.groupby('c').agg({'c': ['count']})
        cluster_df.columns = ['c.count']
        max_c_count = cluster_df['c.count'].max()
        cluster_df.reset_index(inplace=True)
        cluster_df.rename(columns={'index': 'c'})
        cluster_df = cluster_df[(cluster_df['c.count'] > 30) & (cluster_df['c.count'] < max_c_count)]

        if len(cluster_df) == 0:
            return

        multi_ribs_num = len(cluster_df) + 1

        sparse_df = sparse_df[sparse_df['c'].isin(cluster_df['c'].values)]
        thin_line_df = sparse_df.groupby(['c']).agg({'x': ['max']})
        thin_line_df.columns = ['x']
        thin_line_df.reset_index(inplace=True)
        thin_line_df.rename(columns={'index': 'c'}, inplace=True)
        thin_line_point = sparse_df.merge(thin_line_df, on=['c', 'x'], how='inner')

        remove_point_df = sparse_df.merge(thin_line_point, on=['x', 'z'], how='inner')
        remove_point_df.rename(columns={'x': 'x.min'}, inplace=True)

        def make_cartesian(df1=None, df2=None, cartesian_key='cartesian_key'):
            df1[cartesian_key] = 1
            df2[cartesian_key] = 1
            df3 = df1.merge(df2, on=cartesian_key)
            df3.drop([cartesian_key], axis=1, inplace=True)
            return df3

        cartesian_all = make_cartesian(df1=pd.DataFrame({'x': np.arange(x_min, x_max+1, 1)}), df2=remove_point_df)

        cartesian_all = cartesian_all[cartesian_all['x'] >= cartesian_all['x.min']]
        """
        plt.imshow(map2d_image)
        for e in cartesian_all['c'].unique():
            temp_df = cartesian_all[cartesian_all['c'] == e]
            plt.plot(temp_df['y'], temp_df['z'])
        plt.title("{}.map2d_image_thin_lines".format(self.bone_data['c'].unique()[0]))
        plt.show()
        """

        old_class_id = self.bone_data['c'].unique()[0]
        new_bone_data_df = self.bone_data.merge(cartesian_all, on=['x', 'z'], how='left')
        new_bone_data_df = new_bone_data_df[new_bone_data_df['x.min'].isnull()]
        new_bone_data_df.drop(['x.min'], axis=1, inplace=True)

        new_bone_data_3d = sparse_df_to_arr(arr_expected_shape=self.arr_shape, sparse_df=new_bone_data_df)
        new_bone_data_3d_label = skimage.measure.label(new_bone_data_3d, connectivity=2)

        new_bone_data_df, _ = arr_to_sparse_df(label_arr=new_bone_data_3d_label, sort=True, sort_key='c.count',
                                               keep_by_top=True, top_nth=multi_ribs_num)
        new_bone_data_df['c'] = new_bone_data_df['c'].apply(lambda x: "{}-{}".format(old_class_id, x))
        self.bone_data = sparse_df_remove_min(sparse_df=new_bone_data_df, threshold_min=5000)

    def detect_multi_ribs(self):

        _, _, y_min = self.get_basic_axis_feature(feature='min')
        _, _, y_max = self.get_basic_axis_feature(feature='max')

        if len(self.bone_data) < 20000:
            return

        if y_min < self.y_mid_line and y_max < self.y_mid_line:
            rib_min_or_max = 'min'
            shadow_threhold = (y_min + 2*y_max) // 3
        elif y_max > self.y_mid_line and y_min > self.y_mid_line:
            rib_min_or_max = 'max'
            shadow_threhold = (2*y_min + y_max) // 3
        else:
            return

        # print("############ hello")

        map2d_df = self.bone_data.groupby(['y', 'z']).agg({'x': 'sum'})
        map2d_df.reset_index(inplace=True)

        map2d_image = np.ones((self.arr_shape[0], self.arr_shape[2]))
        map2d_image[(map2d_df['z'].values, map2d_df['y'].values)] = 0

        if rib_min_or_max is 'min':
            map2d_image[:, :shadow_threhold] = 0
        else:
            map2d_image[:, shadow_threhold:] = 0

        """
        plt.imshow(map2d_image)
        plt.title("{}.map2d_image".format(self.bone_data['c'].unique()[0]))
        plt.show()
        """

        label_arr = skimage.measure.label(map2d_image, connectivity=2)
        index = label_arr.nonzero()
        sparse_df = pd.DataFrame({'y': index[1],
                                  'z': index[0],
                                  'c': label_arr[index]})
        cluster_df = sparse_df.groupby('c').agg({'c': ['count']})
        cluster_df.columns = ['c.count']
        max_c_count = cluster_df['c.count'].max()
        cluster_df.reset_index(inplace=True)
        cluster_df.rename(columns={'index': 'c'})
        cluster_df = cluster_df[(cluster_df['c.count'] > 10) & (cluster_df['c.count'] < max_c_count)]

        if len(cluster_df) == 0:
            return
        print("############ hello2")
        self.multi_ribs = True

        multi_ribs_num = len(cluster_df) + 1

        sparse_df = sparse_df[sparse_df['c'].isin(cluster_df['c'].values)]
        thin_line_df = sparse_df.groupby(['c', 'y']).agg({'z': ['mean']})
        thin_line_df.columns = ['z']
        thin_line_df['z'] = thin_line_df['z'].apply(lambda x: np.int(x))
        thin_line_df.reset_index(inplace=True)

        if rib_min_or_max is 'min':
            choose_point = thin_line_df.groupby('c').agg({'y': ['max']})
        else:
            choose_point = thin_line_df.groupby('c').agg({'y': ['min']})
        choose_point.columns = ['y']
        choose_point.reset_index(inplace=True)
        choose_point.rename(columns={'index': 'c'})
        thin_point_df = thin_line_df.merge(choose_point, on=['c', 'y'], how='inner')
        thin_point_df.reset_index(inplace=True)
        if rib_min_or_max is 'min':
            thin_point_df.rename(columns={'y': 'y.min'}, inplace=True)
            thin_point_df['y.max'] = y_max
        else:
            thin_point_df.rename(columns={'y': 'y.max'}, inplace=True)
            thin_point_df['y.min'] = y_min

        print("thin_point_df columns :", thin_point_df.columns)

        def make_cartesian(df1=None, df2=None, cartesian_key='cartesian_key'):
            df1[cartesian_key] = 1
            df2[cartesian_key] = 1
            df3 = df1.merge(df2, on=cartesian_key)
            df3.drop([cartesian_key], axis=1, inplace=True)
            return df3

        cartesian_all = make_cartesian(df1=pd.DataFrame({'y': np.arange(y_min, y_max, 1)}), df2=thin_point_df)

        cartesian_all = cartesian_all[(cartesian_all['y'] <= cartesian_all['y.max']) &
                                      (cartesian_all['y'] >= cartesian_all['y.min'])]
        """
        plt.imshow(map2d_image)
        for e in cartesian_all['c'].unique():
            temp_df = cartesian_all[cartesian_all['c'] == e]
            plt.plot(temp_df['y'], temp_df['z'])
        plt.title("{}.map2d_image_thin_lines".format(self.bone_data['c'].unique()[0]))
        plt.show()
        """

        old_class_id = self.bone_data['c'].unique()[0]
        new_bone_data_df = self.bone_data.merge(cartesian_all, on=['y', 'z'], how='left')
        # print("before choose:{},cnt:{}".format(old_class_id, len(new_bone_data_df)))
        new_bone_data_df = new_bone_data_df[new_bone_data_df['y.min'].notnull()]
        # print("after choose:{},cnt:{}".format(old_class_id, len(new_bone_data_df)))
        new_bone_data_df.drop(['y.min', 'y.max'], axis=1, inplace=True)

        new_bone_data_3d = sparse_df_to_arr(arr_expected_shape=self.arr_shape, sparse_df=new_bone_data_df)
        new_bone_data_3d_label = skimage.measure.label(new_bone_data_3d, connectivity=2)

        new_bone_data_df, _ = arr_to_sparse_df(label_arr=new_bone_data_3d_label, sort=True, sort_key='c.count',
                                               keep_by_top=True, top_nth=multi_ribs_num)
        new_bone_data_df['c'] = new_bone_data_df['c'].apply(lambda x: "{}-{}".format(old_class_id, x))
        self.bone_data = new_bone_data_df

    def is_multi_ribs(self):
        # print(self.multi_ribs)
        return self.multi_ribs

    def plot_bone(self, show_all=False, show_3d=False, save=False, save_path=None):
        """
        :param show_all:
        :param show_3d:
        :return:
        """
        img_yzd_arr = np.zeros((self.arr_shape[0], self.arr_shape[1]))
        img_2d_df = self.bone_data.groupby(['y', 'z']).agg({'x': ['count']})
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

    def get_rib_feature_for_predict(self):

        single_bone_feature = {'z_centroid/z_shape': self.get_basic_axis_feature(feature='centroid')[0] / self.arr_shape[0],
                               'x_centroid/x_shape': self.get_basic_axis_feature(feature='centroid')[1] / self.arr_shape[1],
                               'y_centroid/y_shape': self.get_basic_axis_feature(feature='centroid')[2] / self.arr_shape[2],
                               'z_max/z_shape': self.get_basic_axis_feature(feature='max')[0] / self.arr_shape[0],
                               'x_max/x_shape': self.get_basic_axis_feature(feature='max')[1] / self.arr_shape[1],
                               'y_max/y_shape': self.get_basic_axis_feature(feature='max')[2] / self.arr_shape[2],
                               'z_min/z_shape': self.get_basic_axis_feature(feature='min')[0] / self.arr_shape[0],
                               'x_min/x_shape': self.get_basic_axis_feature(feature='min')[1] / self.arr_shape[1],
                               'y_min/y_shape': self.get_basic_axis_feature(feature='min')[2] / self.arr_shape[2],
                               'z_length/z_shape': self.get_basic_axis_feature(feature='length')[0] / self.arr_shape[0],
                               'x_length/x_shape': self.get_basic_axis_feature(feature='length')[1] / self.arr_shape[1],
                               'y_length/y_shape': self.get_basic_axis_feature(feature='length')[2] / self.arr_shape[2],
                               #####
                               'iou_on_xoy': self.get_iou_on_xoy(),
                               'distance_nearest_centroid': self.get_distance_between_centroid_and_nearest_point(),
                               'point_count': len(self.get_bone_data()),
                               #####
                               'mean_z_distance_on_xoy': self.get_features_z_distance_on_xoy()['mean'],
                               'std_z_distance_on_xoy': self.get_features_z_distance_on_xoy()['std'],
                               'std_z_distance_div_mean_z_distance': self.get_features_z_distance_on_xoy()['std_div_mean'],
                               #####
                               'mean_x_distance_on_zoy': self.get_features_x_distance_on_zoy()['mean'],
                               'std_x_distance_on_zoy': self.get_features_x_distance_on_zoy()['std'],
                               'std_x_distance_div_mean_x_distance': self.get_features_x_distance_on_zoy()['std_div_mean'],
                               #####
                               'mean_y_distance_on_zox': self.get_features_y_distance_on_zox()['mean'],
                               'std_y_distance_on_zox': self.get_features_y_distance_on_zox()['std'],
                               'std_y_distance_div_mean_y_distance': self.get_features_y_distance_on_zox()['std_div_mean']
                               }
        return single_bone_feature


if __name__ == "__main__":
    sparse_df = pd.read_csv("/Users/jiangyy/projects/medical-rib/data/ribs_df_cache/135402000404094.csv")
    sparse_df_mult = sparse_df[sparse_df['c'] == 17]

    pass

