import pandas as pd
import numpy as np
from .util import arr_to_sparse_df_only, timer
from deprecated import deprecated


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

        # print(self.sparse_df.columns)

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
        self.sparse_df_y_min_max_for_every_z['y.min'] = self.sparse_df_y_min_max_for_every_z['y.min'].rolling(rolling_window, min_periods=min_periods).min()
        self.sparse_df_y_min_max_for_every_z['y.max'] = self.sparse_df_y_min_max_for_every_z['y.max'].rolling(rolling_window, min_periods=min_periods).max()

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

    def spine_remove_operation(self, value_arr=None):
        _, x_min_max, _ = self.get_min_max_for_every_axis()
        x_min, x_max = x_min_max
        remove_index_y, remove_index_z = self.all_index_in_envelope['y'].values, self.all_index_in_envelope['z'].values
        remove_index_x = np.ones(len(self.all_index_in_envelope), dtype=int) * x_min
        while x_min <= x_max:
            remove_index = remove_index_z, remove_index_x, remove_index_y
            value_arr[remove_index] = 0
            remove_index_x = remove_index_x + 1
            x_min = x_min + 1

    @deprecated(version='3.1.2', reason="elapsed time so long")
    def set_min_max_for_every_xz(self, rolling_window=20, min_periods=10):
        # sparse_df_min_max = self.sparse_df.group by(['x', 'z']).agg({'y': ['min', 'max']}).reset_index(inplace=True).
        # https://pandas.pydata.org/pandas-docs/stable/groupby.html, can support some ideas.
        grouped = self.sparse_df.groupby('x')
        for x_value, group in grouped:
            temp_envelope = pd.DataFrame(index=self.z_all_index)
            yz_envelope_at_x_value = group.groupby('z').agg({'y': ['min', 'max']})
            yz_envelope_at_x_value.columns = ['%s.%s' % e for e in yz_envelope_at_x_value.columns.tolist()]
            temp_envelope = pd.concat([temp_envelope, yz_envelope_at_x_value], axis=1)
            temp_envelope['y.min'] = temp_envelope['y.min'].rolling(rolling_window, min_periods=min_periods).min()
            temp_envelope['y.max'] = temp_envelope['y.max'].rolling(rolling_window, min_periods=min_periods).max()
            temp_envelope.reset_index(inplace=True)
            temp_envelope.rename(columns={'index': 'z'}, inplace=True)
            temp_envelope['x'] = x_value
            self.sparse_df_min_max = self.sparse_df_min_max.append(temp_envelope)

    @deprecated(version='3.1.2', reason="elapsed time so long")
    def get_min_max_for_every_xz(self):
        return self.sparse_df_min_max

    @deprecated(version='3.1.2', reason="elapsed time so long")
    def set_cartesian_product_in_envelope(self):
        z_min_max, x_min_max, y_min_max = self.get_min_max_for_every_axis()
        cartesian_x = pd.DataFrame({'x': np.arange(x_min_max[0], x_min_max[1]+1, 1),
                                    'key': np.zeros(x_min_max[1]-x_min_max[0]+1)})
        cartesian_y = pd.DataFrame({'y': np.arange(y_min_max[0], y_min_max[1]+1, 1),
                                    'key': np.zeros(y_min_max[1]-y_min_max[0]+1)})
        cartesian_z = pd.DataFrame({'z': np.arange(z_min_max[0], z_min_max[1]+1, 1),
                                    'key': np.zeros(z_min_max[1]-z_min_max[0]+1)})
        self.cartesian_all = cartesian_x.merge(cartesian_y.merge(cartesian_z, on='key', how='left'), on='key', how='left')

    @deprecated(version='3.1.2', reason="elapsed time so long")
    def get_cartesian_product_in_envelope(self):
        return self.cartesian_all

    @deprecated(version='3.1.2', reason="elapsed time so long")
    def set_all_pixel_in_envelope(self):
        cartesian_all = self.get_cartesian_product_in_envelope()
        sparse_df_min_max = self.get_min_max_for_every_xz()
        # print(cartesian_all.columns)
        # print(sparse_df_min_max.columns)
        self.pixel_in_envelope = cartesian_all.merge(sparse_df_min_max, on=['x', 'z'], how='left')

        def f(row):
            if row['y'] < row['y.min'] or row['y'] > row['y.max']:
                return False
            return True
        self.pixel_in_envelope['in_envelope'] = self.pixel_in_envelope.apply(lambda row: f(row), axis=1)
        self.pixel_in_envelope = self.pixel_in_envelope[self.pixel_in_envelope['in_envelope']]

    @deprecated(version='3.1.2', reason="elapsed time so long")
    def get_all_pixel_in_envelope(self):
        return self.pixel_in_envelope

    def get_all_index_in_envelope(self):
        return self.pixel_in_envelope['z'].values, self.pixel_in_envelope['x'].values, self.pixel_in_envelope['y'].values

    def set_all_pixel_in_envelope(self):
        z_min_max, x_min_max, y_min_max = self.get_min_max_for_every_axis()
        cartesian_x = pd.DataFrame({'x': np.arange(x_min_max[0], x_min_max[1]+1, 1),
                                    'key': np.zeros(x_min_max[1]-x_min_max[0]+1)})
        cartesian_y = pd.DataFrame({'y': np.arange(y_min_max[0], y_min_max[1]+1, 1),
                                    'key': np.zeros(y_min_max[1]-y_min_max[0]+1)})
        cartesian_z = pd.DataFrame({'z': np.arange(z_min_max[0], z_min_max[1]+1, 1),
                                    'key': np.zeros(z_min_max[1]-z_min_max[0]+1)})
        self.cartesian_all = cartesian_x.merge(cartesian_y.merge(cartesian_z, on='key', how='left'), on='key', how='left')