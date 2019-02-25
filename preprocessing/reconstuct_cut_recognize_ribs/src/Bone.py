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
    Multi_Rib = 4


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

        self.z_mid_line, self.half_z_shape = arr_shape[0] / 2, arr_shape[0] / 2
        self.x_mid_line, self.half_x_shape = arr_shape[1] / 2, arr_shape[1] / 2
        self.y_mid_line, self.half_y_shape = arr_shape[2] / 2, arr_shape[2] / 2

        # deprecated, version 3.1.1
        # self.rib_width = 30
        self.spine_half_width = spine_width / 2
        self.rib_diameter = rib_diameter  # set the threshold of bone's thickness
        # set the y threshold if spine don't combine ribs
        self.spine_connected_rib_y_length_thresholds = spine_connected_rib_y_length_thresholds
        self.through_thresholds = through_thresholds

        # bone type and its subdivision
        self.bone_type = None
        self.rib_type = None
        self.spine_type = None

        # basic feature('min', 'max', 'length', 'centroid') about x,y,z axis
        self.basic_axis_feature = None
        self.set_basic_axis_feature()

        # calc y Max(Max-Min),Mean(Max-Min) group by y, judge whether connected with ribs
        self.y_length_statistics_on_z = None
        self.set_y_length_statistics_on_z()

        # IOU calculation
        self.bone_iou_on_xoy = None
        self.calc_iou_on_xoy()

        # cube volume calculation
        self.bone_volume = None
        self.calc_cube_volume()

        # calc distance between centroid and nearest point.
        self.distance_nearest_centroid = None

        # calc Z Mean(Max-Min) group by (x,y)

        # bone's external cuboid shape
        self.external_cuboid_shape = None

        # center line from binary search in bone prior
        self.prior_zoy_center_y_axis_line_df = prior_zoy_center_y_axis_line_df
        # calc the prob which center line go through bone with, when prob > 0.6, its a spine.
        self.center_line_through_spine_prob = 0.0

        # calc the prob : df in center line +/- spine_half_width.
        self.bone_in_spine_width_prob = 0.0

        # calc sternum
        self.local_centroid_for_sternum = (-1, -1, -1)
        self.local_max_for_sternum = (-1, -1, -1)

        # z length distribution for bone
        self.mean_z_distance_on_xoy = None  # calc Z Mean(Max-Min) group by (x,y)
        self.std_z_distance_on_xoy = None  # calc Z std(Max-Min) group by (x,y)
        self.median_z_distance_on_xoy = None
        self.skew_z_distance_on_xoy = None
        self.kurt_z_distance_on_xoy = None
        self.quantile_down_z_distance_on_xoy = None
        self.quantile_up_z_distance_on_xoy = None
        self.z_distance_std_div_z_distance_mean = None

        # x length distribution for bone
        self.mean_x_distance_on_zoy = None  # calc X Mean(Max-Min) group by (z,y)
        self.std_x_distance_on_zoy = None  # calc X std(Max-Min) group by (z,y)
        self.median_x_distance_on_zoy = None
        self.skew_x_distance_on_zoy = None
        self.kurt_x_distance_on_zoy = None
        self.quantile_down_x_distance_on_zoy = None
        self.quantile_up_x_distance_on_zoy = None
        self.x_distance_std_div_x_distance_mean = None

        # y length distribution for bone
        self.mean_y_distance_on_zox = None  # calc Y Mean(Max-Min) group by (z,x)
        self.std_y_distance_on_zox = None  # calc Y std(Max-Min) group by (z,x)
        self.median_y_distance_on_zox = None
        self.skew_y_distance_on_zox = None
        self.kurt_y_distance_on_zox = None
        self.quantile_down_y_distance_on_zox = None
        self.quantile_up_y_distance_on_zox = None
        self.y_distance_std_div_y_distance_mean = None

        # bone_basic_feature_for_gbdt_calculate_initial
        # partial initial
        self.selective_initial()

    def selective_initial(self):
        if self.detection_objective == 'rib':
            # calc distance between centroid and nearest point.
            self.calc_distance_between_centroid_and_nearest_point()

            # calc Z Max(Max-Min) group by (x,y)
            self.set_features_z_distance_on_xoy()
            self.set_features_x_distance_on_zoy()
            self.set_features_y_distance_on_zox()

        else:

            # speed up not spine or sternum judgement
            if self.judge_not_spine_or_sternum_speedup():
                self.bone_type = BoneType.OtherBig_Bone
                return
            # calc Z Max(Max-Min) group by (x,y)
            self.set_features_z_distance_on_xoy()
            self.set_features_x_distance_on_zoy()
            self.set_features_y_distance_on_zox()

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
    def set_features_z_distance_on_xoy(self):
        z_distance = self.bone_data.groupby(['x', 'y']).agg({'z': ['max', 'min']})
        z_distance.columns = ['%s.%s' % e for e in z_distance.columns.tolist()]
        z_distance['z_distance'] = z_distance['z.max'] - z_distance['z.min']
        self.mean_z_distance_on_xoy = z_distance['z_distance'].mean()
        self.std_z_distance_on_xoy = z_distance['z_distance'].std()
        self.median_z_distance_on_xoy = z_distance['z_distance'].median()
        self.skew_z_distance_on_xoy = z_distance['z_distance'].skew()
        self.kurt_z_distance_on_xoy = z_distance['z_distance'].kurt()
        self.quantile_down_z_distance_on_xoy = z_distance['z_distance'].quantile(0.25)
        self.quantile_up_z_distance_on_xoy = z_distance['z_distance'].quantile(0.75)
        self.z_distance_std_div_z_distance_mean = (self.std_z_distance_on_xoy + 0.001) / (self.mean_z_distance_on_xoy + 0.001)
        del z_distance

    def set_features_x_distance_on_zoy(self):
        x_distance = self.bone_data.groupby(['z', 'y']).agg({'x': ['max', 'min']})
        x_distance.columns = ['%s.%s' % e for e in x_distance.columns.tolist()]
        x_distance['x_distance'] = x_distance['x.max'] - x_distance['x.min']
        self.mean_x_distance_on_zoy = x_distance['x_distance'].mean()
        self.std_x_distance_on_zoy = x_distance['x_distance'].std()
        self.median_x_distance_on_zoy = x_distance['x_distance'].median()
        self.skew_x_distance_on_zoy = x_distance['x_distance'].skew()
        self.kurt_x_distance_on_zoy = x_distance['x_distance'].kurt()
        self.quantile_down_x_distance_on_zoy = x_distance['x_distance'].quantile(0.25)
        self.quantile_up_x_distance_on_zoy = x_distance['x_distance'].quantile(0.75)
        self.x_distance_std_div_x_distance_mean = (self.std_x_distance_on_zoy + 0.001) / (self.mean_x_distance_on_zoy + 0.001)
        del x_distance

    def set_features_y_distance_on_zox(self):
        y_distance = self.bone_data.groupby(['z', 'x']).agg({'y': ['max', 'min']})
        y_distance.columns = ['%s.%s' % e for e in y_distance.columns.tolist()]
        y_distance['y_distance'] = y_distance['y.max'] - y_distance['y.min']
        self.mean_y_distance_on_zox = y_distance['y_distance'].mean()
        self.std_y_distance_on_zox = y_distance['y_distance'].std()
        self.median_y_distance_on_zox = y_distance['y_distance'].median()
        self.skew_y_distance_on_zox = y_distance['y_distance'].skew()
        self.kurt_y_distance_on_zox = y_distance['y_distance'].kurt()
        self.quantile_down_y_distance_on_zox = y_distance['y_distance'].quantile(0.25)
        self.quantile_up_y_distance_on_zox = y_distance['y_distance'].quantile(0.75)
        self.y_distance_std_div_y_distance_mean = (self.std_y_distance_on_zox + 0.001) / (self.mean_y_distance_on_zox + 0.001)
        del y_distance

    def get_features_z_distance_on_xoy(self):
        features_z_distance_on_xoy = {'mean': self.mean_z_distance_on_xoy,
                                      'std': self.std_z_distance_on_xoy,
                                      'median': self.median_z_distance_on_xoy,
                                      'skew': self.skew_z_distance_on_xoy,
                                      'kurt': self.kurt_z_distance_on_xoy,
                                      'quantile_down': self.quantile_down_z_distance_on_xoy,
                                      'quantile_up': self.quantile_up_z_distance_on_xoy,
                                      'std_div_mean': self.z_distance_std_div_z_distance_mean
                                      }
        return features_z_distance_on_xoy

    def get_features_x_distance_on_zoy(self):
        features_x_distance_on_zoy = {'mean': self.mean_x_distance_on_zoy,
                                      'std': self.std_x_distance_on_zoy,
                                      'median': self.median_x_distance_on_zoy,
                                      'skew': self.skew_x_distance_on_zoy,
                                      'kurt': self.kurt_x_distance_on_zoy,
                                      'quantile_down': self.quantile_down_x_distance_on_zoy,
                                      'quantile_up': self.quantile_up_x_distance_on_zoy,
                                      'std_div_mean': self.x_distance_std_div_x_distance_mean
                                      }
        return features_x_distance_on_zoy

    def get_features_y_distance_on_zox(self):
        features_y_distance_on_zox = {'mean': self.mean_y_distance_on_zox,
                                      'std': self.std_y_distance_on_zox,
                                      'median': self.median_y_distance_on_zox,
                                      'skew': self.skew_y_distance_on_zox,
                                      'kurt': self.kurt_y_distance_on_zox,
                                      'quantile_down': self.quantile_down_y_distance_on_zox,
                                      'quantile_up': self.quantile_up_y_distance_on_zox,
                                      'std_div_mean': self.y_distance_std_div_y_distance_mean
                                      }
        return features_y_distance_on_zox

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
    """
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

        if self.get_features_z_distance_on_xoy()['median'] > self.rib_diameter:
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

        if self.get_features_z_distance_on_xoy()['median'] > self.rib_diameter:
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

    def is_multi_rib(self):
        if self.get_iou_on_xoy() >= 0.45:
            return False

        if len(self.get_bone_data()) < 40000:
            return False

        if self.get_distance_between_centroid_and_nearest_point() < 20:
            return False

        max_nonzero_internal = self.detect_multi_ribs()
        if max_nonzero_internal < 5:
            return False

        return True
    """
    def detect_multi_ribs(self):
        z_distribution = self.get_bone_data()[(self.get_bone_data()['x'] >
                                              self.get_basic_axis_feature(feature='centroid')[1]-3)
                                              & (self.get_bone_data()['x'] <
                                              self.get_basic_axis_feature(feature='centroid')[1]+3)].groupby('z').agg({'y': 'count'})
        z_distribution.columns = ['count']
        z_distribution.reset_index(inplace=True)
        z_distribution.rename({'index': 'z'})
        z_index = list(z_distribution[z_distribution['count'] > 0]['z'])

        if len(z_index) > 1:
            max_nonzero_internal = max(z_index[i+1] - z_index[i] for i in range(len(z_index)-1))
        else:
            max_nonzero_internal = 0

        del z_distribution
        return max_nonzero_internal

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

        if self.get_basic_axis_feature(feature='length')[0] < self.half_z_shape:
            return False

        # if not self.center_line_through_bone(through_thresholds=0.6):
            # return False
        return True

    def is_half_spine(self):
        # _, x_centroid, _ = self.get_basic_axis_feature(feature='centroid')
        _, x_local_centroid, _ = self.local_centroid_for_sternum
        if x_local_centroid < self.x_mid_line:
            return False

        if self.get_basic_axis_feature(feature='max')[0] > self.arr_shape[0] - 5:
            return False

        # if not self.center_line_through_bone(through_thresholds=0.6):
            # return False

        if self.get_basic_axis_feature(feature='length')[2] / self.get_basic_axis_feature(feature='length')[0] > 5:
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
        if self.get_features_z_distance_on_xoy()['median'] < 50:
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
    """
    def is_up_bone(self):
        # to do
        z_max, _, y_max = self.get_basic_axis_feature(feature='max')
        if z_max > self.arr_shape[0] - 50:
            return True
        _, _, y_min = self.get_basic_axis_feature(feature='min')
        if y_min < 20 or y_max > self.arr_shape[2] - 20:
            return True
        return False
    """

    def detect_spine_and_sternum(self):
        """
        :return:
        """
        if self.bone_type is not None:
            return

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
                               'iou_on_xoy': self.get_iou_on_xoy(),
                               'distance_nearest_centroid': self.get_distance_between_centroid_and_nearest_point(),
                               'point_count': len(self.get_bone_data()),
                               'mean_z_distance_on_xoy': self.get_features_z_distance_on_xoy()['mean'],
                               'std_z_distance_on_xoy': self.get_features_z_distance_on_xoy()['std'],
                               'std_z_distance_div_mean_z_distance': self.get_features_z_distance_on_xoy()['std_div_mean'],
                               'median_z_distance_on_xoy': self.get_features_z_distance_on_xoy()['median'],
                               'skew_z_distance_on_xoy': self.get_features_z_distance_on_xoy()['skew'],
                               'kurt_z_distance_on_xoy': self.get_features_z_distance_on_xoy()['kurt'],
                               'quantile_down_z_distance_on_xoy': self.get_features_z_distance_on_xoy()['quantile_down'],
                               'quantile_up_z_distance_on_xoy': self.get_features_z_distance_on_xoy()['quantile_up'],
                               'mean_x_distance_on_zoy': self.get_features_x_distance_on_zoy()['mean'],
                               'std_x_distance_on_zoy': self.get_features_x_distance_on_zoy()['std'],
                               'median_x_distance_on_zoy': self.get_features_x_distance_on_zoy()['median'],
                               'std_x_distance_div_mean_x_distance': self.get_features_x_distance_on_zoy()['std_div_mean'],
                               'skew_x_distance_on_zoy': self.get_features_x_distance_on_zoy()['skew'],
                               'kurt_x_distance_on_zoy': self.get_features_x_distance_on_zoy()['kurt'],
                               'quantile_down_x_distance_on_zoy': self.get_features_x_distance_on_zoy()['quantile_down'],
                               'quantile_up_x_distance_on_zoy': self.get_features_x_distance_on_zoy()['quantile_up'],
                               'mean_y_distance_on_zox': self.get_features_y_distance_on_zox()['mean'],
                               'std_y_distance_on_zox': self.get_features_y_distance_on_zox()['std'],
                               'median_y_distance_on_zox': self.get_features_y_distance_on_zox()['median'],
                               'std_y_distance_div_mean_y_distance': self.get_features_y_distance_on_zox()['std_div_mean'],
                               'skew_y_distance_on_zox': self.get_features_y_distance_on_zox()['skew'],
                               'kurt_y_distance_on_zox': self.get_features_y_distance_on_zox()['kurt'],
                               'quantile_down_y_distance_on_zox': self.get_features_y_distance_on_zox()['quantile_down'],
                               'quantile_up_y_distance_on_zox': self.get_features_y_distance_on_zox()['quantile_up'],
                               'max_nonzero_internal': self.detect_multi_ribs()
                               }
        return single_bone_feature

    def print_bone_info(self):
        print('#'*30 + ' bone ' + '#' * 30)
        print('bone direction:', self.left_or_right())
        print('bone Type:', self.bone_type)
        print('rib type:', self.rib_type)
        print('iou = %.2f' % self.get_iou_on_xoy())
        print('z_distance_max is', self.get_features_z_distance_on_xoy())
        print('nearest point is', self.get_distance_between_centroid_and_nearest_point())
        print('through center line rate = {}'.format(self.center_line_through_spine_prob))
