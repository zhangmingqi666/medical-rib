from enum import Enum
from .util import *
import numpy as np
from deprecated import deprecated
from interval import Interval
import warnings
warnings.filterwarnings('ignore')


class BoneType(Enum):
    Fragment_Bone = 1
    Rib_Bone = 2
    OtherBig_Bone = 3
    Spine_Bone = 4
    Sternum_Bone = 5


class RibType(Enum):
    High_Circle_Rib = 1
    Mid_Curved_Rib = 2
    Low_Straight_Rib = 3


class SpineType(Enum):
    Only_Spine_Complete = 1
    Half_Spine = 2
    Frag_Spine = 3
    Spine_Connected_rib = 4


class Direction(Enum):
    Left = 0
    Right = 1


class Bone:

    def __init__(self, bone_data=None, arr_shape=None, spine_width=100, rib_diameter=30,
                 spine_connected_rib_y_length_thresholds=180, through_thresholds=0.6,
                 prior_zoy_center_y_axis_line_df=None, detection_objective='rib'):
        # DataFrame
        self.bone_data = bone_data
        self.arr_shape = arr_shape

        self.detection_objective = detection_objective
        # deprecated, version 3.1.1
        # self.rib_spine_joint_point = []

        self.z_mid_line, self.half_z_shape = arr_shape[0] / 2, arr_shape[0] / 2
        self.y_mid_line, self.half_y_shape = arr_shape[2] / 2, arr_shape[2] / 2
        self.x_mid_line, self.half_x_shape = arr_shape[1] / 2, arr_shape[1] / 2

        # deprecated, version 3.1.1
        # self.rib_width = 30
        self.spine_half_width = spine_width / 2
        self.rib_diameter = rib_diameter
        self.spine_connected_rib_y_length_thresholds = spine_connected_rib_y_length_thresholds
        self.through_thresholds = through_thresholds

        # bone type and its subdivision
        self.bone_type = None
        self.rib_type = None
        self.spine_type = None

        # basic feature about x,y,z axis
        self.bone_data_min = None
        self.bone_data_max = None
        self.bone_data_length = None
        self.centroid = None
        self.basic_axis_feature = None
        self.set_basic_axis_feature()

        # calc y Max(Max-Min),Mean(Max-Min) group by y, judge whether connected with ribs
        self.y_length_statistics_on_z = None
        self.set_y_length_statistics_on_z()
        # self.y_max_length_on_z = None
        # self.y_mean_length_on_z = None
        # self.y_std_length_on_z = None

        # IOU calculation
        self.bone_iou_on_xoy = None
        self.calc_iou_on_xoy()

        # cube volume calculation
        self.bone_volume = None
        self.calc_cube_volume()

        # calc distance between centroid and nearest point.
        self.distance_nearest_centroid = None

        # calc Z Max(Max-Min) group by (x,y)
        self.max_z_distance_on_xoy = None

        # deprecated, version 3.1.1
        # High circle rib.
        self.rib_spine_joint_point = []
        self.rib_sternum_joint_point = []

        # bone's external cuboid shape
        self.external_cuboid_shape = None
        # self.set_external_cuboid_shape()

        # center line from binary search in bone prior
        self.prior_zoy_center_y_axis_line_df = prior_zoy_center_y_axis_line_df
        # calc the prob which center line go through bone with, when prob > 0.6, its a spine.
        self.center_line_through_spine_prob = 0.0

        # calc the prob : df in center line +/- spine_half_width.
        self.bone_in_spine_width_prob = 0.0

        # calc sternum
        self.local_centroid_for_sternum = (-1, -1, -1)
        self.local_max_for_sternum = (-1, -1, -1)

        # partial initial
        self.selective_initial()

    def selective_initial(self):
        if self.detection_objective == 'rib':
            # calc distance between centroid and nearest point.
            self.calc_distance_between_centroid_and_nearest_point()

            # calc Z Max(Max-Min) group by (x,y)
            self.set_max_z_distance_on_xoy()

        else:

            # speed up not spine or sternum judgement
            if self.judge_not_spine_or_sternum_speedup():
                self.bone_type = BoneType.OtherBig_Bone
                return
            # calc Z Max(Max-Min) group by (x,y)
            self.set_max_z_distance_on_xoy()

            # calc the prob which center line go through bone with, when prob > 0.6, its a spine.
            self.set_center_line_through_spine_prob()

            # calc the prob : df in center line +/- spine_half_width.
            self.calc_prob_bone_in_spine_width()

            # calc used for judging spine connected to ribs
            self.set_y_length_statistics_on_z()

            # calc local centroid for sternum
            self.set_local_centroid_for_sternum()

    def get_bone_data(self):
        return self.bone_data

    def set_basic_axis_feature(self):
        self.bone_data_min = self.bone_data['z'].min(), self.bone_data['x'].min(), self.bone_data['y'].min()
        self.bone_data_max = self.bone_data['z'].max(), self.bone_data['x'].max(), self.bone_data['y'].max()
        self.bone_data_length = tuple(np.array(self.bone_data_max) - np.array(self.bone_data_min) + 1)
        self.centroid = self.bone_data['z'].mean(), self.bone_data['x'].mean(), self.bone_data['y'].mean()
        self.basic_axis_feature = {'min': self.bone_data_min,
                                   'max': self.bone_data_max,
                                   'length': self.bone_data_length,
                                   'centroid': self.centroid}

    def get_basic_axis_feature(self, feature='centroid'):
        assert feature in self.basic_axis_feature
        return self.basic_axis_feature.get(feature)

    """
    used for calc whether some ribs connected with spine.
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
        # self.bone_data['dis_nearest'] = ((self.bone_data['z'] - z_centroid)**2 +
        #                                 (self.bone_data['x'] - x_centroid)**2 +
        #                                 (self.bone_data['y'] - y_centroid)**2)
        # self.bone_data['dis_nearest'] = self.bone_data['dis_nearest'].apply(lambda x: np.sqrt(x))
        # self.bone_data.loc[:, 'dis_nearest'] = self.bone_data['dis_nearest'].apply(lambda x: np.sqrt(x))
        self.distance_nearest_centroid = self.bone_data['dis_nearest'].min()

    def get_distance_between_centroid_and_nearest_point(self):
        return self.distance_nearest_centroid

    def left_or_right(self):
        _, _, y_centroid = self.get_basic_axis_feature(feature='centroid')
        if y_centroid > self.y_mid_line:
            return Direction.Right
        return Direction.Left

    """
        restrict rib height for every xoy
    """
    def set_max_z_distance_on_xoy(self):
        z_distance = self.bone_data.groupby(['x', 'y']).agg({'z': ['max', 'min']})
        z_distance.columns = ['%s.%s' % e for e in z_distance.columns.tolist()]
        z_distance['z_distance'] = z_distance['z.max'] - z_distance['z.min']
        self.max_z_distance_on_xoy = z_distance['z_distance'].max()
        del z_distance

    def get_max_z_distance_on_xoy(self):
        return self.max_z_distance_on_xoy

    def calc_iou_on_xoy(self):
        pipeline_area = len(self.bone_data.groupby(['x', 'y']))
        _, x_length, y_length = self.get_basic_axis_feature(feature='length')
        rectangle_area = x_length * y_length
        self.bone_iou_on_xoy = 1.0 * pipeline_area / rectangle_area

    def get_iou_on_xoy(self):
        return self.bone_iou_on_xoy

    def calc_cube_volume(self):
        z_length, x_length, y_length = self.get_basic_axis_feature(feature='length')
        self.bone_volume = z_length * x_length * y_length

    def get_cube_volume(self):
        return self.bone_volume

    def set_external_cuboid_shape(self):
        self.external_cuboid_shape = (self.get_basic_axis_feature()['length'][0],
                                      self.get_basic_axis_feature()['length'][1],
                                      self.get_basic_axis_feature()['length'][2])

    def get_external_cuboid_shape(self):
        return self.external_cuboid_shape

    def calc_external_cuboid_arr(self):
        z_min, x_min, y_min = self.get_basic_axis_feature(feature='min')

        def f(row):
            row['z'] = row['z'] - z_min
            row['x'] = row['x'] - x_min
            row['y'] = row['y'] - y_min
            return row

        external_cuboid_shape = self.get_external_cuboid_shape()
        temp_df = self.bone_data.apply(lambda row: f(row), axis=1)
        # print(temp_df['z'].min(), temp_df['x'].min(), temp_df['y'].min())
        external_cuboid_arr = sparse_df_to_arr(arr_expected_shape=external_cuboid_shape, sparse_df=temp_df)
        del temp_df
        gc.collect()
        return external_cuboid_arr

    def is_first_rib(self):
        if self.get_iou_on_xoy() >= 0.38:
            return False

        _, _, y_centroid = self.get_basic_axis_feature(feature='centroid')
        # cannot deviate far away from the center line.
        if abs(y_centroid - self.y_mid_line) > 40:
                return False

        if self.get_distance_between_centroid_and_nearest_point() < 20:
            return False

        return True

    def is_mid_rib(self):
        if self.get_iou_on_xoy() >= 0.23:
            return False

        if self.get_distance_between_centroid_and_nearest_point() < 10:
            return False

        if self.get_max_z_distance_on_xoy() > self.rib_diameter:
            return False

        z_centroid, _, _ = self.get_basic_axis_feature(feature='centroid')
        _, _, y_max = self.get_basic_axis_feature(feature='max')
        _, _, y_min = self.get_basic_axis_feature(feature='min')

        if self.left_or_right() == Direction.Right:
            if self.bone_data[self.bone_data['y'] == y_min]['z'].mean() < z_centroid:
                return False
        else:
            if self.bone_data[self.bone_data['y'] == y_max]['z'].mean() < z_centroid:
                return False
        return True

    def is_low_rib(self):

        if self.get_max_z_distance_on_xoy() > self.rib_diameter:
            return False

        # prior distriction

        z_centroid, _, _ = self.get_basic_axis_feature(feature='centroid')
        _, _, y_max = self.get_basic_axis_feature(feature='max')
        _, _, y_min = self.get_basic_axis_feature(feature='min')

        if self.left_or_right() == Direction.Right:
            if self.bone_data[self.bone_data['y'] == y_min]['z'].mean() < z_centroid:
                return False
        else:
            if self.bone_data[self.bone_data['y'] == y_max]['z'].mean() < z_centroid:
                return False

        if self.get_cube_volume() < 100000:
            return False
        return True

    def is_rib(self):
        return self.bone_type == BoneType.Rib_Bone

    def cut_first_rib(self):
        z_centroid, x_centroid, y_centroid = self.get_basic_axis_feature(feature='centroid')
        df_1st_quadrant = self.bone_data[(self.bone_data['x'] >= x_centroid) & (self.bone_data['y'] >= y_centroid)]
        df_2nd_quadrant = self.bone_data[(self.bone_data['x'] < x_centroid) & (self.bone_data['y'] >= y_centroid)]
        df_3rd_quadrant = self.bone_data[(self.bone_data['x'] < x_centroid) & (self.bone_data['y'] < y_centroid)]
        df_4th_quadrant = self.bone_data[(self.bone_data['x'] >= x_centroid) & (self.bone_data['y'] < y_centroid)]

        joint_point_1st = df_1st_quadrant[df_1st_quadrant['y'] == df_1st_quadrant['y'].min()]
        joint_point_2nd = df_2nd_quadrant[df_2nd_quadrant['y'] == df_1st_quadrant['y'].min()]
        joint_point_4th = df_4th_quadrant[df_4th_quadrant['y'] == df_4th_quadrant['y'].max()]
        joint_point_3rd = df_3rd_quadrant[df_3rd_quadrant['y'] == df_4th_quadrant['y'].max()]

        self.rib_spine_joint_point = [(joint_point_1st['z'].iloc[0], joint_point_1st['x'].iloc[0], joint_point_1st['y'].iloc[0]),
                                      (joint_point_4th['z'].iloc[0], joint_point_4th['x'].iloc[0], joint_point_4th['y'].iloc[0])]

        self.rib_sternum_joint_point = [(joint_point_2nd['z'].iloc[0], joint_point_2nd['x'].iloc[0], joint_point_2nd['y'].iloc[0]),
                                        (joint_point_3rd['z'].iloc[0], joint_point_3rd['x'].iloc[0], joint_point_3rd['y'].iloc[0])]

    def cut_other_rib(self):

        _, _, y_max = self.get_basic_axis_feature(feature='max')
        _, _, y_min = self.get_basic_axis_feature(feature='min')
        if self.left_or_right() == Direction.Right:
            temp_df = self.bone_data[self.bone_data['y'] == y_min]
            self.rib_spine_joint_point = [(temp_df['z'].iloc[0], temp_df['x'].iloc[0], temp_df['y'].iloc[0])]
        else:
            temp_df = self.bone_data[self.bone_data['y'] == y_max]
            self.rib_spine_joint_point = [(temp_df['z'].iloc[0], temp_df['x'].iloc[0], temp_df['y'].iloc[0])]
        # self.rib_sternum_joint_point = []

    def set_center_line_through_spine_prob(self):
        """
        base on center line in zoy depending on bone prior class,
        here, we count the probability center line through bone.
        """
        # spine_body_statistics = pd.DataFrame({'z': self.prior_zoy_center_y_axis_line_df['z'],
        #                                       'y.center': self.prior_zoy_center_y_axis_line_df[:, 1]})

        y_min_max_grpby_z = self.bone_data.groupby('z').agg({'y': ['min', 'max']})
        y_min_max_grpby_z.columns = ['%s.%s' % e for e in y_min_max_grpby_z.columns.tolist()]
        y_min_max_grpby_z.reset_index(inplace=True)
        y_min_max_grpby_z.rename(columns={'index': 'z'})

        spine_body_statistics = self.prior_zoy_center_y_axis_line_df.merge(y_min_max_grpby_z, on='z', how='inner')
        # center line through spine
        spine_body_statistics['through'] = spine_body_statistics.apply(lambda row: row['y.center'] in Interval(row['y.min'], row['y.max']), axis=1)

        self.center_line_through_spine_prob = 1.0 * spine_body_statistics['through'].sum() / len(spine_body_statistics)

    def get_center_line_through_spine_prob(self):
        return self.center_line_through_spine_prob

    def center_line_through_bone(self, through_thresholds=0.6):
        return self.get_center_line_through_spine_prob() >= through_thresholds

    def is_complete_spine(self):

        # _, x_centroid, _ = self.get_basic_axis_feature(feature='centroid')
        _, x_local_centroid, _ = self.local_centroid_for_sternum
        if x_local_centroid < self.x_mid_line:
            return False

        if self.get_max_z_distance_on_xoy() < self.half_z_shape:
            return False

        if not self.center_line_through_bone(through_thresholds=0.6):
            return False
        return True

    def is_half_spine(self):
        # _, x_centroid, _ = self.get_basic_axis_feature(feature='centroid')
        _, x_local_centroid, _ = self.local_centroid_for_sternum
        if x_local_centroid < self.x_mid_line:
            return False

        if not self.center_line_through_bone(through_thresholds=0.6):
            return False

        return True

    def is_spine(self):
        return self.bone_type == BoneType.Spine_Bone

    def calc_prob_bone_in_spine_width(self):
        intersection_df = self.bone_data.merge(self.prior_zoy_center_y_axis_line_df, on='z', how='left')

        def f(row):
            if abs(row['y'] - row['y.center']) < self.spine_half_width:
                return True
            return False
        intersection_df['in'] = intersection_df.apply(lambda row: f(row), axis=1)
        self.bone_in_spine_width_prob = 1.0 * intersection_df[intersection_df['in']].count() / len(intersection_df)

    def get_prob_bone_in_spine_width(self):
        return self.bone_in_spine_width_prob

    def is_fragment_spine(self):
        if self.get_prob_bone_in_spine_width() < 0.5:
            return False
        return True

    def spine_connected_rib(self):
        return self.get_y_length_statistics_on_z(feature='max') > self.spine_connected_rib_y_length_thresholds

    def set_local_centroid_for_sternum(self):
        local_df = self.bone_data[(self.bone_data['y'] > self.y_mid_line - 60) &
                                  (self.bone_data['y'] < self.y_mid_line + 60)]
        if len(local_df) < 1000:
            return
        self.local_centroid_for_sternum = local_df['z'].mean(), local_df['x'].mean(), local_df['y'].mean()
        self.local_max_for_sternum = local_df['z'].max(), local_df['x'].max(), local_df['y'].max()
        del local_df

    def get_local_centroid_for_sternum(self):
        return self.local_centroid_for_sternum

    def is_complete_sternum(self):
        # _, x_centroid, y_centroid = self.get_basic_axis_feature(feature='centroid')
        # if x_centroid > self.x_mid_line:
        #    return False
        if self.local_centroid_for_sternum is None:
            return False

        z_centroid, x_centroid, y_centroid = self.local_centroid_for_sternum
        z_max, x_max, _ = self.local_max_for_sternum
        if x_centroid > self.x_mid_line + 50:
            return False

        if z_max > self.arr_shape[0] - 10:
            return False

        """
        if self.get_max_z_distance_on_xoy() < 50:
            return False
        """
        # if self.get_basic_axis_feature(feature='length')[2] / self.get_basic_axis_feature(feature='length')[0] > 4:
        #    return False
        return True

    def is_sternum(self):
        return self.bone_type == BoneType.Sternum_Bone

    def judge_not_spine_or_sternum_speedup(self):
        z_centroid, _, y_centroid = self.get_basic_axis_feature(feature='centroid')
        temp_df = self.prior_zoy_center_y_axis_line_df[(self.prior_zoy_center_y_axis_line_df['z'] > z_centroid - 5) &
                                                       (self.prior_zoy_center_y_axis_line_df['z'] < z_centroid + 5)]
        y_mid = temp_df['y.center'].mean()
        if abs(y_mid - y_centroid) > 100:
            return True
        return False

    def is_skull(self):
        _, _, y_centroid = self.get_basic_axis_feature(feature='centroid')
        if abs(y_centroid - self.y_mid_line) > 50:
            return False

        z_max, _, _ = self.get_basic_axis_feature(feature='max')
        if z_max < self.arr_shape[0] - 5:
            return False
        return True

    def is_up_bone(self):
        # todo
        z_max, _, y_max = self.get_basic_axis_feature(feature='max')
        if z_max > self.arr_shape[0] - 50:
            return True
        _, _, y_min = self.get_basic_axis_feature(feature='min')
        if y_min < 20 or y_max > self.arr_shape[2] - 20:
            return True
        return False

    @deprecated(version='3.1.1', reason="all in morphology")
    def get_rib_spine_joint_point(self):
        return self.rib_spine_joint_point

    @deprecated(version='3.1.1', reason="all in morphology")
    def get_rib_sternum_joint_point(self):
        return self.rib_sternum_joint_point

    @deprecated(version='3.1.1', reason="all in morphology")
    def get_joint_point(self):
        print(type(self.rib_spine_joint_point))
        temp_list = self.get_rib_spine_joint_point()
        if self.is_fisrt_rib():
            pass
            # temp_list.extend(self.get_rib_sternum_joint_point())
        return temp_list

    def set_bone_type(self):
        """
        :return:
        """
        if self.bone_type is not None:
            return

        if self.detection_objective == 'rib':
            if self.is_first_rib():
                self.bone_type = BoneType.Rib_Bone
                self.rib_type = RibType.High_Circle_Rib
                # self.cut_first_rib()
                return

            if self.is_mid_rib():
                self.bone_type = BoneType.Rib_Bone
                self.rib_type = RibType.Mid_Curved_Rib
                # self.cut_other_rib()
                return

            if self.is_low_rib():
                self.bone_type = BoneType.Rib_Bone
                self.rib_type = RibType.Low_Straight_Rib
                # self.cut_other_rib()
                return

        else:
            if self.is_complete_spine():
                self.bone_type = BoneType.Spine_Bone
                self.spine_type = SpineType.Only_Spine_Complete
                return

            if self.is_half_spine():
                self.bone_type = BoneType.Spine_Bone
                self.spine_type = SpineType.Half_Spine
                return

            if self.is_complete_sternum():
                self.bone_type = BoneType.Sternum_Bone
                return

        self.bone_type = BoneType.OtherBig_Bone

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
        """
        if show_all:
            img_arr = sparse_df_to_arr(arr_expected_shape=self.arr_shape, sparse_df=self.bone_data)
        else:
            img_arr = self.calc_external_cuboid_arr()

        if show_3d:
            plot_3d(img_arr, threshold=0.5)
        else:
            plt.imshow(img_arr.sum(axis=1))
        plt.show()
        del img_arr
        gc.collect()
        """

    def print_bone_info(self):
        print('#'*30 + ' bone ' + '#' * 30)
        print('bone direction:', self.left_or_right())
        print('bone Type:', self.bone_type)
        print('rib type:', self.rib_type)
        print('iou = %.2f' % self.get_iou_on_xoy())
        print('z_distance_max is', self.set_max_z_distance_on_xoy())
        print('nearest point is', self.get_distance_between_centroid_and_nearest_point())
        print('through center line rate = {}'.format(self.center_line_through_spine_prob))
