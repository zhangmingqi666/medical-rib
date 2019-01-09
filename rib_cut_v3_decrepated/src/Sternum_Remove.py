import pandas as pd
import numpy as np
from .util import arr_to_sparse_df_only
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .util import arr_to_sparse_df_only, timer
from deprecated import deprecated


class SternumRemove:
    def __init__(self, bone_data_arr=None, bone_data_df=None, bone_data_shape=[],
                 allow_envelope_expand=False, expand_width=20):
        self.sparse_df = None
        if bone_data_arr is not None:
            self.sparse_df = arr_to_sparse_df_only(bone_data_arr)
        elif bone_data_df is not None:
            self.sparse_df = bone_data_df
        self.bone_data_shape = bone_data_shape

        self.allow_envelope_expand = allow_envelope_expand
        self.expand_width = expand_width

        """
        self.z_all_index = range(self.bone_data_shape[0])
        self.sparse_df_min_max = pd.DataFrame({})
        with timer('_________calc min max envelope for every xz'):
            self.set_min_max_for_every_xz(rolling_window=20, min_periods=10)
        """

        """
        The below 3 variable used for cartesian dot
        """
        self.min_max_for_z = None
        self.min_max_for_x = None
        self.min_max_for_y = None
        self.set_min_max_for_every_axis()

        """
        # deprecated @version 3.1.2, reason: such a long elapsed time.
        # 作笛卡尔积之后,join 获得 所有包络线内的像素.
        self.cartesian_all = None
        with timer('_________calc cartesian product in envelope'):
            self.set_cartesian_product_in_envelope()

        self.pixel_in_envelope = None
        with timer('_________calc intersect between cartesian product and envelope'):
            self.set_all_pixel_in_envelope()
        """

        self.sparse_df_y_min_max_for_every_z = None
        self.all_index_in_envelope = None
        self.set_y_min_max_for_every_z()

    def set_min_max_for_every_axis(self):
        self.min_max_for_x = self.sparse_df['x'].min(), self.sparse_df['x'].max()
        self.min_max_for_y = self.sparse_df['y'].min(), self.sparse_df['y'].max()
        self.min_max_for_z = self.sparse_df['z'].min(), self.sparse_df['z'].max()

    def get_min_max_for_every_axis(self):
        return self.min_max_for_z, self.min_max_for_x, self.min_max_for_y

    def set_y_min_max_for_every_z(self, rolling_window=20, min_periods=10):
        self.sparse_df_y_min_max_for_every_z = self.sparse_df.groupby('z').agg({'y': ['min', 'max']})
        self.sparse_df_y_min_max_for_every_z.columns = ['%s.%s' % e for e in self.sparse_df_y_min_max_for_every_z.columns]
        self.sparse_df_y_min_max_for_every_z.reset_index(inplace=True)
        self.sparse_df_y_min_max_for_every_z.rename(columns={'index': 'z'})
        self.sparse_df_y_min_max_for_every_z['y.min'] = self.sparse_df_y_min_max_for_every_z['y.min'].rolling(rolling_window, min_periods=min_periods).max()
        self.sparse_df_y_min_max_for_every_z['y.max'] = self.sparse_df_y_min_max_for_every_z['y.max'].rolling(rolling_window, min_periods=min_periods).min()

        if self.allow_envelope_expand:
            self.sparse_df_y_min_max_for_every_z['y.min'] = self.sparse_df_y_min_max_for_every_z['y.min'] - self.expand_width
            self.sparse_df_y_min_max_for_every_z['y.max'] = self.sparse_df_y_min_max_for_every_z['y.max'] + self.expand_width

        z_min_max, _, y_min_max = self.get_min_max_for_every_axis()
        cartesian_y = pd.DataFrame({'y': np.arange(y_min_max[0], y_min_max[1] + 1, 1),
                                    'key': np.zeros(y_min_max[1] - y_min_max[0] + 1)})
        cartesian_z = pd.DataFrame({'z': np.arange(z_min_max[0], z_min_max[1] + 1, 1),
                                    'key': np.zeros(z_min_max[1] - z_min_max[0] + 1)})
        cartesian_all = cartesian_y.merge(cartesian_z, on='key', how='left')

        all_index_df_in_envelope = cartesian_all.merge(self.sparse_df_y_min_max_for_every_z, on='z')
        self.all_index_in_envelope = all_index_df_in_envelope[(all_index_df_in_envelope['y'] <= all_index_df_in_envelope['y.max']) & (all_index_df_in_envelope['y'] >= all_index_df_in_envelope['y.min'])]
        del all_index_df_in_envelope

    def sternum_remove_operation(self, value_arr=None):
        _, x_min_max, _ = self.get_min_max_for_every_axis()
        x_min, x_max = x_min_max
        remove_index_y, remove_index_z = self.all_index_in_envelope['y'].values, self.all_index_in_envelope['z'].values
        remove_index_x = np.ones(len(self.all_index_in_envelope), dtype=int) * x_min
        while x_min <= x_max:
            remove_index = remove_index_z, remove_index_x, remove_index_y
            value_arr[remove_index] = 0
            remove_index_x = remove_index_x + 1
            x_min = x_min + 1

    def sternum_connect_ribs(self):
        y_length = self.sparse_df['y'].max() - self.sparse_df['y'].min()
        if y_length < 150:
            return False
        return True

    def cut_sternum_v2(self):
        if self.sternum_connect_ribs():
            y_line = self.sparse_df[self.sparse_df['x'] < self.bone_data_shape[1] / 2].groupby('y').agg({'z': 'count'})
            y_line.columns = ['z.count']
            y_line.reset_index(inplace=True)
            y_line.rename(columns={'index': 'y'})
            y_line['z.count'] = y_line['z.count'].rolling(10, min_periods=1, center=True).max()
            plt.plot(y_line['y'], y_line['z.count'])
            plt.show()


    """
    def sternum_connect_ribs(self, sternum_bone=Bone()):
        if sternum_bone.get_basic_axis_feature()['length'][2] < 150:
            return False
        return True

    
    def cut_sternum(self, sternum_bone=Bone(), data_arr=None):
        if sternum_connect_ribs(sternum_bone):
            max_z_length_group_by_y = sternum_bone[sternum_bone['x'] < sternum_bone.arr_shape[1]] \
                .bone_data.groupby('y').agg({'z': ['max', 'min']})
            max_z_length_group_by_y.columns = ['%s.%s' % e for e in max_z_length_group_by_y.columns.tolist()]
            max_z_length_group_by_y.reset_index(inplace=True)
            max_z_length_group_by_y.rename(columns={'index': 'y'})
            max_z_length_group_by_y['z.length'] = max_z_length_group_by_y['z.max'] - max_z_length_group_by_y['z.min']
            y_left = max_z_length_group_by_y[max_z_length_group_by_y['y'] > sternum_bone.arr_shape[2] / 3 &
                                             max_z_length_group_by_y['y'] < sternum_bone.arr_shape[2] / 2]['z'][
                'z.length'].min()
            y_right = max_z_length_group_by_y[max_z_length_group_by_y['y'] < 2 * sternum_bone.arr_shape[2] / 3 &
                                              max_z_length_group_by_y['y'] > sternum_bone.arr_shape[2] / 2][
                'z.length'].min()

            data_arr[:, :data_arr.shape / 2, y_left:y_right] = 0
        else:
            data_arr[sternum_bone.get_basic_axis_feature()['min'][0]:sternum_bone.get_basic_axis_feature()['max'][0],
            sternum_bone.get_basic_axis_feature()['min'][1]:sternum_bone.get_basic_axis_feature()['max'][1],
            sternum_bone.get_basic_axis_feature()['min'][2]:sternum_bone.get_basic_axis_feature()['max'][2]] = 0
        """

"""
class SternumRemove:
    def __init__(self, bone_data_arr=None, bone_data_df=None, bone_data_shape=[]):
        self.sparse_df = None
        if bone_data_arr is not None:
            self.sparse_df = arr_to_sparse_df_only(bone_data_arr)
        elif bone_data_df is not None:
            self.sparse_df = bone_data_df

        self.envelope = None
        self.set_min_max_envelope()

    def set_min_max_envelope(self):
        self.envelope = self.sparse_df.groupby('z').agg({'y': ['min', 'max']})
        self.envelope.columns = ['%s.%s' % e for e in self.envelope.columns.tolist()]
        self.envelope.reset_index(inplace=True)
        self.envelope.rename(columns={'index': 'z'})
        self.sparse_df = self.sparse_df.merge(self.envelope, on='z')

        def f(row):
            if row['y'] < row['y.min'] or row['y'] > row['y.max']:
                return False
            return True
        self.sparse_df['in'] = self.sparse_df.apply(lambda row: f(row), axis=1)
        self.sparse_df = self.sparse_df[self.sparse_df['in']]

    def get_all_index_in_envelope(self):
        return self.sparse_df['z'].values, self.sparse_df['x'].values, self.sparse_df['y'].values
"""