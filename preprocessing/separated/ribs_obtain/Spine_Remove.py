import pandas as pd
import numpy as np
# from projects
import sys, os


def add_python_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_python_path(os.getcwd())
from preprocessing.separated.ribs_obtain.util import arr_to_sparse_df_only


class SpineRemove:
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

        def railings_interpolate(spine_remaining_df=None,
                                 spine_width=150):
            _y_center_df = spine_remaining_df.groupby('z').agg({'y': ['mean', 'min', 'max']})
            _y_center_df.columns = ['%s.%s' % e for e in _y_center_df.columns]
            _y_center_df.reset_index(inplace=True)
            _y_center_df.rename(columns={'index': 'z'}, inplace=True)
            _y_center_df['min'] = _y_center_df.apply(lambda row: max(row['y.mean']-spine_width, row['y.min']), axis=1)
            _y_center_df['max'] = _y_center_df.apply(lambda row: min(row['y.mean']+spine_width, row['y.max']), axis=1)

            return _y_center_df

        def relu_interpolate(section_df, full_length=self.bone_data_shape[0], margin_min=5, margin_max=20,
                             is_expand=self.allow_envelope_expand, expand_width=self.expand_width):

            # get spine width for every z
            section_df['y.length'] = section_df.apply(lambda row: row['y.max']-row['y.min'], axis=1)
            average_y_length = section_df['y.length'].mean()

            # when a rib connected spine, the width near the rib will be greater than others.
            section_df[(section_df['y.length'] < average_y_length - margin_min) |
                       (section_df['y.length'] > average_y_length + margin_max)] = None
            full_df = pd.DataFrame(index=[i for i in range(full_length)])

            # interpolate the location for connected ribs.
            full_df['y.min'] = pd.Series(section_df['y.min'].values, index=section_df['y.min'].index)
            full_df['y.max'] = pd.Series(section_df['y.max'].values, index=section_df['y.max'].index)
            full_df.interpolate(inplace=True)
            full_df = full_df.fillna(method='ffill').fillna(method='bfill')
            full_df.reset_index(inplace=True)
            full_df.rename(columns={'index': 'z'}, inplace=True)
            if is_expand:
                full_df['y.min'] = full_df['y.min'] - expand_width
                full_df['y.max'] = full_df['y.max'] + expand_width

            # print("shape: ", self.bone_data_shape[0])
            # print("len: ", len(full_df['y.min']))

            return full_df

        center_df = railings_interpolate(spine_remaining_df=self.sparse_df, spine_width=150)
        _relu_score = relu_interpolate(section_df=center_df, margin_min=5, margin_max=50, is_expand=False)
        z_min_max, _, y_min_max = self.get_min_max_for_every_axis()
        cartesian_y = pd.DataFrame({'y': np.arange(y_min_max[0], y_min_max[1] + 1, 1),
                                    'key': np.zeros(y_min_max[1] - y_min_max[0] + 1)})
        cartesian_z = pd.DataFrame({'z': np.arange(0, self.bone_data_shape[0], 1),
                                    'key': np.zeros(self.bone_data_shape[0])})
        cartesian_all = cartesian_y.merge(cartesian_z, on='key', how='left')

        all_index_df_in_envelope = cartesian_all.merge(_relu_score, on='z')
        self.all_index_in_envelope = all_index_df_in_envelope[(all_index_df_in_envelope['y'] <= all_index_df_in_envelope['y.max'])
                                                              & (all_index_df_in_envelope['y'] >= all_index_df_in_envelope['y.min'])]

        del all_index_df_in_envelope

    def get_all_index_in_envelope(self):
        return self.all_index_in_envelope

    def spine_remove_operation(self, value_arr=None):
        _, x_min_max, _ = self.get_min_max_for_every_axis()
        x_min, x_max = x_min_max

        x_min = x_min - 100
        x_max = self.bone_data_shape[1] - 1

        remove_index_y, remove_index_z = self.all_index_in_envelope['y'].values, self.all_index_in_envelope['z'].values
        remove_index_x = np.ones(len(self.all_index_in_envelope), dtype=int) * x_min

        while x_min <= x_max:
            remove_index = remove_index_z, remove_index_x, remove_index_y
            value_arr[remove_index] = 0
            remove_index_x = remove_index_x + 1
            x_min = x_min + 1
