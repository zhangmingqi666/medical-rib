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

    def set_y_min_max_for_every_z(self, rolling_window=20, min_periods=10, apart_distance=100):

        def railings_interpolate(spine_remaining_df=None,
                                 spine_width=None):
            _y_center_df = spine_remaining_df.groupby('z').agg({'y': ['mean', 'min', 'max', 'std']})
            _y_center_df.columns = ['%s.%s' % e for e in _y_center_df.columns]
            _y_center_df.reset_index(inplace=True)
            _y_center_df.rename(columns={'index': 'z'}, inplace=True)
            _y_center_df['min'] = _y_center_df.apply(lambda row: max(row['y.mean']-row['y.std'], row['y.min']), axis=1)
            _y_center_df['max'] = _y_center_df.apply(lambda row: min(row['y.mean']+row['y.std'], row['y.max']), axis=1)

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

        def railings_relu_interpolate(spine_remaining_df=None):
            # groupby y.statistics by z, y.std Equivalent to radius @ z
            _y_center_df = spine_remaining_df.groupby('z').agg({'y': ['mean', 'min', 'max', 'std']})
            _y_center_df.columns = ['%s.%s' % e for e in _y_center_df.columns]

            ### add missing index:
            #_y_center_df = pd.concat((_y_center_df, pd.DataFrame(index=[i for i in range(self.bone_data_shape[0])])),
            #                         axis=1)

            y_std_mean, y_std_std = _y_center_df['y.std'].mean(), _y_center_df['y.std'].std()
            _y_center_df['y.std.distribution'] = _y_center_df['y.std'].apply(
                lambda x: np.abs(x - y_std_mean) / y_std_std)
            _y_center_df[_y_center_df['y.std.distribution'] > 1.0][['y.min', 'y.max']] = np.NAN

            _y_center_df[['y.min', 'y.max']] = _y_center_df[['y.min', 'y.max']].interpolate()
            _y_center_df[['y.min', 'y.max']] = _y_center_df[['y.min', 'y.max']].fillna(method='ffill').fillna(
                method='bfill')
            _y_center_df.reset_index(inplace=True)
            _y_center_df.rename(columns={'index': 'z'})
            _y_center_df['y.mean'] = _y_center_df['y.mean'].rolling(30).mean().fillna(method='ffill').fillna(
                method='bfill')

            return _y_center_df

        def nan_cave_relu_interpolate(section_df, margin_min=50, expand_width=20):
            section_df['y.length'] = section_df.apply(lambda row: row['y.max'] - row['y.min'], axis=1)
            average_y_length = section_df['y.length'].mean()
            # when a rib connected spine, the width near the rib will be greater than others.
            section_df[(section_df['y.length'] < average_y_length - margin_min)] = None
            # section_df[(section_df['y.length'] > average_y_length + margin_min)] = None    # 长的肋骨
            section_df = section_df.merge(pd.DataFrame({'z': [i for i in range(self.bone_data_shape[0])]}),
                                          on='z', how='outer')

            section_df[['y.min', 'y.max']] = section_df[['y.min', 'y.max']].\
                interpolate().\
                fillna(method='ffill').\
                fillna(method='bfill')


            #section_df.to_csv("/Users/jiangyy/Desktop/hehehe.csv", index=False)

            section_df['y.min'] = section_df.apply(lambda row: row['y.min'] - (expand_width if row['y.length'] is None
                                                                              else 0), axis=1)
            section_df['y.max'] = section_df.apply(lambda row: row['y.max'] + (expand_width if row['y.length'] is None
                                                                               else 0), axis=1)

            # 长的肋骨
            # section_df['y.min'] = section_df.apply(lambda row: row['y.min'] + (expand_width if row['y.length'] == 1
            #                                                                   else 0), axis=1)
            # section_df['y.max'] = section_df.apply(lambda row: row['y.max'] - (expand_width if row['y.length'] == 1
            #                                                                   else 0), axis=1)

            # section_df['y.max'] = section_df['y.max'].rolling(15).min().fillna(method='ffill').fillna(
            #   method='bfill')

            # section_df.to_csv("/home/wangshuli/Desktop/1.csv", index=False)
            section_df = section_df.dropna(subset=['z'])
            section_df.sort_values('z', inplace=True)
            return section_df

        def resize_long_y(section_df, margin_max=40):
            series_y_maxMin = section_df['y.max'].rolling(15).min()
            series_y_maxMax = section_df['y.max'].rolling(15).max()
            series_y_minMin = section_df['y.min'].rolling(15).min()
            series_y_minMax = section_df['y.min'].rolling(15).max()

            df_y_max_maxmin = pd.DataFrame({'y.max_min':series_y_maxMin,'y.max_max':series_y_maxMax,'y.min_min':series_y_minMin,'y.min_max':series_y_minMax})
            # df_y_max_maxmin.to_csv('/home/wangshuli/Desktop/roll.csv')
            df_y_max_maxmin[['y.max_min', 'y.max_max','y.min_min','y.min_max']] = df_y_max_maxmin[['y.max_min', 'y.max_max','y.min_min','y.min_max']]. \
                interpolate(). \
                fillna(method='ffill'). \
                fillna(method='bfill')
            section_df = section_df.join(df_y_max_maxmin)
            section_df['y.gap_right'] = section_df.apply(lambda row: row['y.max_max'] - row['y.max_min'], axis=1)
            section_df['y.gap_left'] = section_df.apply(lambda row: row['y.min_max'] - row['y.min_min'], axis=1)

            # section_df.to_csv("/home/wangshuli/Desktop/2.csv", index=False)

            section_df['y.max'] = section_df.apply(
                lambda row: row['y.max'] if row['y.gap_right'] < margin_max else row['y.max_min'], axis=1)
            section_df['y.min'] = section_df.apply(
                lambda row: row['y.min'] if row['y.gap_left'] < margin_max else row['y.min_max'], axis=1)

            # section_df.to_csv("/home/wangshuli/Desktop/3.csv", index=False)
            return section_df

        y_mean = self.sparse_df['y'].mean()
        y_left, y_right = y_mean - apart_distance, y_mean + apart_distance    # apart_distance = 100
        apart_spine_remaining_df = self.sparse_df[(self.sparse_df['y'] > y_left) &
                                                  (self.sparse_df['y'] < y_right)]
        y_center_df = railings_relu_interpolate(spine_remaining_df=apart_spine_remaining_df)

        temp_df = apart_spine_remaining_df.merge(y_center_df, on='z', how='inner')
        temp_df['inner'] = temp_df.apply(lambda row: np.abs(row['y'] - row['y.mean']) / row['y.std'], axis=1)
        apart_spine_remaining_df_enhanced = temp_df[temp_df['inner'] < 3.0]
        y_center_df_enhanced = railings_relu_interpolate(spine_remaining_df=apart_spine_remaining_df_enhanced)

        y_center_df_enhanced_caved = nan_cave_relu_interpolate(y_center_df_enhanced)
        y_center_df_enhanced_caved = resize_long_y(y_center_df_enhanced_caved)

        # center_df = railings_interpolate(spine_remaining_df=apart_spine_remaining_df_enhanced, spine_width=150)
        # _relu_score = relu_interpolate(section_df=center_df, margin_min=5, margin_max=50, is_expand=False)
        z_min_max, _, y_min_max = self.get_min_max_for_every_axis()

        def make_cartesian(df1=None, df2=None, cartesian_key='cartesian_key'):
            df1[cartesian_key] = 1
            df2[cartesian_key] = 1
            df3 = df1.merge(df2, on=cartesian_key)
            df3.drop([cartesian_key], axis=1, inplace=True)
            return df3

        cartesian_all = make_cartesian(df1=pd.DataFrame({'y': np.arange(y_min_max[0], y_min_max[1] + 1, 1)}),
                                       df2=pd.DataFrame({'z': np.arange(0, self.bone_data_shape[0], 1)}))

        """
        cartesian_y = pd.DataFrame({'y': np.arange(y_min_max[0], y_min_max[1] + 1, 1),
                                    'key': np.zeros(y_min_max[1] - y_min_max[0] + 1)})
        cartesian_z = pd.DataFrame({'z': np.arange(0, self.bone_data_shape[0], 1),
                                    'key': np.zeros(self.bone_data_shape[0])})
        cartesian_all = cartesian_y.merge(cartesian_z, on='key', how='left')
        """

        all_index_df_in_envelope = cartesian_all.merge(y_center_df_enhanced_caved, on='z')
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
