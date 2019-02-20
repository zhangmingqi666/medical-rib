import numpy as np
import pandas as pd


class SternumRemove:
    def __init__(self, bone_data_arr=None, bone_data_df=None, bone_data_shape=None, center_line=None,
                 hu_threshold=400, width=100, output_prefix=None):

        self.value_arr = bone_data_arr
        self.sternum_df = bone_data_df
        self.arr_shape = bone_data_shape

        self.width = width
        self.hu_threshold = hu_threshold
        self.center_line = center_line

        self.left_envelope_line = None
        self.right_envelope_line = None

        self.set_sternum_envelope_line()

    def set_sternum_envelope_line(self):
        """
        calculate the sternum envelope line
        """
        y_limit_min = self.center_line - self.width
        y_limit_max = self.center_line + self.width
        self.sternum_df = self.sternum_df[(self.sternum_df['y'] > y_limit_min) & (self.sternum_df['y'] < y_limit_max)]
        sternum_df_group_by_z = self.sternum_df.groupby('z').agg({'y': ['min', 'max']})
        sternum_df_group_by_z.columns = ['%s.%s' % e for e in sternum_df_group_by_z.columns.tolist()]
        sternum_df_group_by_z.reset_index(inplace=True)
        sternum_df_group_by_z.rename(columns={'index': 'z'})

        sternum_df_group_by_z['y.min'] = sternum_df_group_by_z['y.min'].apply(lambda x: None if x < y_limit_min + 10 else x)
        sternum_df_group_by_z['y.max'] = sternum_df_group_by_z['y.max'].apply(lambda x: None if x > y_limit_max - 10 else x)

        # sternum_df_group_by_z.fillna(method='bfill')

        self.left_envelope_line = np.ones(self.arr_shape[0]) * self.center_line
        self.right_envelope_line = np.ones(self.arr_shape[0]) * self.center_line

        index = sternum_df_group_by_z['z'].values

        self.left_envelope_line[index] = sternum_df_group_by_z['y.min'].values
        self.right_envelope_line[index] = sternum_df_group_by_z['y.max'].values

        self.left_envelope_line[self.left_envelope_line > self.center_line] = None
        # y_envelope_min[y_envelope_min < y_limit_min] = None
        self.right_envelope_line[self.right_envelope_line < self.center_line] = None
        # y_envelope_max[y_envelope_max > y_limit_max] = None
        # y_envelope_min = pd.Series(y_envelope_min).fillna(method='bfill').rolling(5, min_periods=1).max()
        # y_envelope_max = pd.Series(y_envelope_max).fillna(method='bfill').rolling(5, min_periods=1).min()
        self.left_envelope_line = pd.Series(self.left_envelope_line).fillna(method='bfill').fillna(method='ffill').rolling(15, min_periods=5).mean()
        self.right_envelope_line = pd.Series(self.right_envelope_line).fillna(method='bfill').fillna(method='ffill').rolling(15, min_periods=5).mean()

        self.left_envelope_line.apply(lambda x: x-10)
        self.right_envelope_line.apply(lambda x: x+10)
        """
        plt.figure()
        plt.plot(y_envelope_min, np.arange(arr_shape[0]))
        plt.plot(y_envelope_max, np.arange(arr_shape[0]))
        plt.savefig('{}sternum_envelope.png'.format(self.output_prefix))
        """
        del sternum_df_group_by_z

    def sternum_remove_operation(self, value_arr=None):
        """
        split sternum by using sternum envelope lines
        """
        # z_index = np.arange(len(sternum_envelope_line_left), dtype=np.int16)
        # x_index = np.ones(len(sternum_envelope_line_left), dtype=np.int16)
        y_left_index = np.array(self.left_envelope_line, dtype=np.int16)
        y_right_index = np.array(self.right_envelope_line, dtype=np.int16)

        z_index = []
        y_index = []
        for i in range(len(self.left_envelope_line)):
            y_index_temp = range(y_left_index[i], y_right_index[i]+1)
            y_index.extend(y_index_temp)
            z_index.extend([i for j in range(len(y_index_temp))])

        x_index = np.ones(len(z_index), dtype=np.int16)
        y_index = np.array(y_index, dtype=np.int16)
        z_index = np.array(z_index, dtype=np.int16)
        for i in range(3*self.arr_shape[1]//5):
            value_arr[z_index, i * x_index, y_index] = 0

        del y_left_index
        del y_right_index
