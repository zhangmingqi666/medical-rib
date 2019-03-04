from enum import Enum
import matplotlib.pyplot as plt
import gc
import numpy as np
from interval import Interval
import warnings
import sys, os


def add_python_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_python_path(os.getcwd())
# from projects
from preprocessing.separated.ribs_obtain.util import sparse_df_to_arr
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


class BoneSpine:

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

        self.prior_zoy_center_y_axis_line_df = prior_zoy_center_y_axis_line_df

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

        # calc distance between centroid and nearest point.
        self.distance_nearest_centroid = None

        self.local_centroid = None
        self.set_local_centroid()
        # speed up not spine or sternum judgement
        if self.judge_not_spine_or_sternum_speedup():
            self.bone_type = BoneType.OtherBig_Bone
            return

        # calc the prob which center line go through bone with, when prob > 0.6, its a spine.
        # self.set_center_line_through_spine_prob()

        # calc the prob : df in center line +/- spine_half_width.
        # self.calc_prob_bone_in_spine_width()

        # calc used for judging spine connected to ribs
        self.y_length_statistics_on_z = None
        self.set_y_length_statistics_on_z()

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
        _, x_local_centroid, _ = self.local_centroid
        if x_local_centroid < self.x_mid_line:
            return False

        if self.get_basic_axis_feature(feature='length')[0] < self.half_z_shape:
            return False

        # if not self.center_line_through_bone(through_thresholds=0.6):
            # return False
        return True

    def is_half_spine(self):
        # _, x_centroid, _ = self.get_basic_axis_feature(feature='centroid')
        _, x_local_centroid, _ = self.local_centroid
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

    """
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
    """

    # def is_fragment_spine(self):
    #    if self.get_prob_bone_in_spine_width() < 0.5:
    #        return False
    #    return True

    def spine_connected_rib(self):
        return self.get_y_length_statistics_on_z(feature='max') > self.spine_connected_rib_y_length_thresholds
    
    def set_local_centroid(self):
        local_df = self.bone_data[(self.bone_data['y'] > self.y_mid_line - 60) &
                                  (self.bone_data['y'] < self.y_mid_line + 60)]
        if len(local_df) < 1000:
            return
        self.local_centroid = local_df['z'].mean(), local_df['x'].mean(), local_df['y'].mean()
        del local_df

    def get_local_centroid(self):
        return self.local_centroid

    def judge_not_spine_or_sternum_speedup(self):
        z_centroid, _, y_centroid = self.get_basic_axis_feature(feature='centroid')
        temp_df = self.prior_zoy_center_y_axis_line_df[(self.prior_zoy_center_y_axis_line_df['z'] > z_centroid - 5) &
                                                       (self.prior_zoy_center_y_axis_line_df['z'] < z_centroid + 5)]
        y_mid = temp_df['y.center'].mean()
        if abs(y_mid - y_centroid) > 100:
            return True
        return False

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

        self.bone_type = BoneType.OtherBig_Bone




