import numpy as np

from enum import Enum


class BoneType(Enum):
    Fragment_Bone = 0
    Rib_Bone = 1
    OtherBig_Bone = 2


class RibType(Enum):
    High_Circle_Rib = 0
    Mid_Curved_Rib = 1
    Low_Straight_Rib = 2


class Direction(Enum):
    Left = 0
    Right = 1


class Bone:
    
    def __init__(self, bone_data=None, shape=None, naturally_shedding=False):
        self.bone_data = bone_data
        self.shape = shape
        self.rib_spine_joint_point = []

        self.z_mid_line = shape[0] / 2
        self.y_mid_line = shape[2] / 2

        self.rib_width = 30

        self.bone_type = None
        self.rib_type = None

        self.naturallyShedding = naturally_shedding

        self.centroid = self.bone_data['z'].mean(), self.bone_data['x'].mean(), self.bone_data['y'].mean()

        # calc Z Max(Max-Min) group by (x,y)
        self.z_max_distance = None
        self.set_max_z_distance()

        # IOU calculation
        self.bone_iou = None
        self.calc_iou()

        # cube volume calculation
        self.bone_volume = None
        self.calc_cube_volume()

        # calc distance between centroid and nearest point.
        self.distance_nearest_centroid = None
        self.calc_distance_between_centroid_and_nearest_point()

        # High circle rib.
        self.rib_sternum_joint_point = []

    ###################################################################################################################

    def get_centroid(self):
        return self.centroid

    def get_bone_data(self):
        return self.bone_data

    def calc_distance_between_centroid_and_nearest_point(self):
        z_centroid, x_centroid, y_centroid = self.get_centroid()
        
        def f(row):
            return np.sqrt((row['z']-z_centroid)**2 + (row['x']-x_centroid)**2 + (row['y']-y_centroid)**2)
        self.bone_data['dis_nearest'] = self.bone_data.apply(lambda row: f(row), axis=1)
        self.distance_nearest_centroid = self.bone_data['dis_nearest'].min()

    def get_distance_between_centroid_and_nearest_point(self):
        return self.distance_nearest_centroid

    def left_or_right(self):
        if self.bone_data['y'].mean() > self.shape[2]/2:
            return Direction.Right
        return Direction.Left

    def set_max_z_distance(self):
        z_distance = self.bone_data.groupby(['x', 'y']).agg({'z': ['max', 'min']})
        z_distance.columns = ['%s.%s' % e for e in z_distance.columns.tolist()]
        z_distance['z_distance'] = z_distance['z.max']-z_distance['z.min']
        self.z_max_distance = z_distance['z_distance'].max()

    def get_max_z_distance(self):
        return self.z_max_distance

    def get_iou(self):
        return self.bone_iou

    def calc_iou(self):
        pipeline_area = len(self.bone_data.groupby(['x', 'y']))
        rectangle_area = ((self.bone_data['x'].max() - self.bone_data['x'].min() + 1) 
                          * (self.bone_data['y'].max() - self.bone_data['y'].min() + 1))
        self.bone_iou = 1.0 * pipeline_area / rectangle_area

    def calc_cube_volume(self):
        self.bone_volume = ((self.bone_data['x'].max() - self.bone_data['x'].min()) *
                            (self.bone_data['y'].max() - self.bone_data['y'].min()) * 
                            (self.bone_data['z'].max() - self.bone_data['z'].min()))

    def get_cube_volume(self):
        return self.bone_volume

    def is_fisrt_rib(self):
        if self.get_iou() >= 0.38: 
            return False

        _, _, y_centroid = self.get_centroid()
        # cannot deviate far away from the center line.
        if abs(y_centroid - self.y_mid_line) > 40:
                return False

        if self.get_distance_between_centroid_and_nearest_point() < 20: 
            return False

        return True

    def is_mid_rib(self):
        if self.get_iou() >= 0.23:
            return False

        if self.get_distance_between_centroid_and_nearest_point() < 10:
            return False

        if self.get_max_z_distance() > self.rib_width:
            return False

        z_centroid, _, _ = self.get_centroid()
        if self.left_or_right() == Direction.Right:
            if self.bone_data[self.bone_data['y'] == self.bone_data['y'].min()]['z'].mean() < z_centroid:
                return False
        else:
            if self.bone_data[self.bone_data['y'] == self.bone_data['y'].max()]['z'].mean() < z_centroid:
                return False
        return True

    def is_low_rib(self):

        if self.get_max_z_distance() > self.rib_width:
            return False

        # prior distriction

        z_centroid, _, _ = self.get_centroid()
        if self.left_or_right() == Direction.Right:
            if self.bone_data[self.bone_data['y'] == self.bone_data['y'].min()]['z'].mean() < z_centroid:
                return False
        else:
            if self.bone_data[self.bone_data['y'] == self.bone_data['y'].max()]['z'].mean() < z_centroid:
                return False

        if self.get_cube_volume() < 100000:
            return False
        return True

    def is_rib(self):
        return self.bone_type == BoneType.Rib_Bone

    def cut_first_rib(self):
        z_centroid, x_centroid, y_centroid = self.get_centroid()
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
        if self.left_or_right() == Direction.Right:
            temp_df = self.bone_data[self.bone_data['y'] == self.bone_data['y'].min()]
            self.rib_spine_joint_point = [(temp_df['z'].iloc[0], temp_df['x'].iloc[0], temp_df['y'].iloc[0])]
        else:
            temp_df = self.bone_data[self.bone_data['y'] == self.bone_data['y'].max()]
            self.rib_spine_joint_point = [(temp_df['z'].iloc[0], temp_df['x'].iloc[0], temp_df['y'].iloc[0])]
        # self.rib_sternum_joint_point = []

    def get_rib_spine_joint_point(self):
        return self.rib_spine_joint_point

    def get_rib_sternum_joint_point(self):
        return self.rib_sternum_joint_point

    def get_joint_point(self):
        print(type(self.rib_spine_joint_point))
        temp_list = self.get_rib_spine_joint_point()
        if self.is_fisrt_rib():
            pass
            # temp_list.extend(self.get_rib_sternum_joint_point())
        return temp_list

    def set_bone_type(self):

        if self.is_fisrt_rib():
            self.bone_type = BoneType.Rib_Bone
            self.rib_type = RibType.High_Circle_Rib
            self.cut_first_rib()
            return

        if self.is_mid_rib():
            self.bone_type = BoneType.Rib_Bone
            self.rib_type = RibType.Mid_Curved_Rib
            self.cut_other_rib()
            return

        if self.is_low_rib():
            self.bone_type = BoneType.Rib_Bone
            self.rib_type = RibType.Low_Straight_Rib
            self.cut_other_rib()
            return

        # if self.get_max_z_distance() >= self.rib_width:
        self.bone_type = BoneType.OtherBig_Bone

    def print_bone_info(self):
        print('#'*30 + ' bone ' + '#' * 30)
        print('bone direction:', self.left_or_right())
        print('bone Type:', self.bone_type)
        print('rib type:', self.rib_type)
        print('iou = %.2f' % self.get_iou())
        print('Z_distance_max is', self.get_max_z_distance())
        print('nearest point is', self.get_distance_between_centroid_and_nearest_point())
