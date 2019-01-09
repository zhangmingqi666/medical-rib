import numpy as np
import gc
import pandas as pd
from .util import arr_to_sparse_df
import matplotlib.pyplot as plt
import skimage
from .util import *


class BonePrior:
    """
    Assumed that everyone is strict symmetrical object, including bones,
    """
    def __init__(self, binary_arr=None):
        """
        :param binary_arr: input only binary arr
        """
        self.binary_arr = binary_arr

        # decrepated
        # self.binary_arr[self.binary_arr < hu_threshold] = 0
        # self.binary_arr[self.binary_arr >= hu_threshold] = 1

        # calc zoy-axis center line.
        self.zoy_symmetric_y_axis_line = None
        self.zoy_symmetric_y_axis_line_df = None
        self.set_zoy_center_y_axis_line()

        self.arr_shape = self.binary_arr.shape

        self.spine_y_width = 0.0

        # will not use
        del self.binary_arr
        gc.collect()

    def set_zoy_center_y_axis_line(self):
        zoy_count_distribution = self.binary_arr.sum(axis=1)
        y_shape = zoy_count_distribution.shape[1]
        zoy_count_distribution[:, :(y_shape//3)] = 0
        zoy_count_distribution[:, (y_shape * 2 // 3):] = 0
        zoy_acc_distribution = np.add.accumulate(zoy_count_distribution, axis=1)

        def numpy_arr_binary_search(arr):
            center_line_arr = np.zeros((arr.shape[0], 2))
            for i in range(arr.shape[0]):
                left_idx = np.searchsorted(arr[i], arr[i, -1] / 2.0, side='left')
                right_idx = np.searchsorted(arr[i], arr[i, -1] / 2.0, side='right')
                center_line_arr[i] = i, (left_idx + right_idx) // 2
            return center_line_arr

        zoy_symmetric_y_axis_line = numpy_arr_binary_search(zoy_acc_distribution)
        zoy_symmetric_y_axis_line_df = pd.DataFrame({'z': zoy_symmetric_y_axis_line[:, 0],
                                                     'y.center': zoy_symmetric_y_axis_line[:, 1]})

        zoy_symmetric_y_axis_line_df['filter_mean'] = zoy_symmetric_y_axis_line_df['y.center'].rolling(100,
                                                                                                       min_periods=50,
                                                                                                       win_type='parzen',
                                                                                                       axis=0).mean()
        zoy_symmetric_y_axis_line_df['filter_mean'] = zoy_symmetric_y_axis_line_df['filter_mean'].fillna(method='bfill', axis=0).fillna(method='ffill', axis=0)

        def f(row):
            if abs(row['y.center'] - row['filter_mean']) > 70:
                return None
            return row['y.center']
        zoy_symmetric_y_axis_line_df['y.center'] = zoy_symmetric_y_axis_line_df.apply(lambda row: f(row), axis=1)
        zoy_symmetric_y_axis_line_df['y.center'] = zoy_symmetric_y_axis_line_df['y.center'].interpolate()
        self.zoy_symmetric_y_axis_line_df = zoy_symmetric_y_axis_line_df

    def get_zoy_symmetric_y_axis_line_df(self):
        return self.zoy_symmetric_y_axis_line_df

    def get_prior_shape(self):
        return self.arr_shape

    def set_zoy_width(self):
        pass

    def get_zoy_width(self):
        pass


def calc_sternum_center_line(value_arr, hu_threshold=400):
    """
    calculate the center line with 0.5 square
    :param value_arr: source CT array
    :param hu_threshold: HU threshold
    :return:
    """
    # arr[arr.shape[0]//2:arr.shape[0], 0:(arr.shape[1]//3), [(arr.shape[2]//2-60), (arr.shape[2]//2+60)]] = 0
    # arr =

    binary_arr = value_arr.copy()
    binary_arr[binary_arr < hu_threshold] = 0
    binary_arr[binary_arr >= hu_threshold] = 1

    y_line_arr = binary_arr[:, 0:binary_arr.shape[1]//2, :].sum(axis=1).sum(axis=0)
    # ax1 = plt.subplot(211)
    # plt.imshow(binary_arr[:, 0:binary_arr.shape[1]//2, :].sum(axis=1))
    del binary_arr

    y_line_arr[:len(y_line_arr)//4] = 0
    y_line_arr[3*len(y_line_arr)//4:] = 0
    y_line_arr = np.add.accumulate(y_line_arr)

    def numpy_arr_binary_search(arr):
        left_idx = np.searchsorted(arr, arr[-1] / 2.0, side='left')
        right_idx = np.searchsorted(arr, arr[-1] / 2.0, side='right')
        center_line_arr = (left_idx + right_idx) // 2
        return center_line_arr

    y_line_center = numpy_arr_binary_search(y_line_arr)
    # print("胸骨切割中心线：", y_line_center)
    # plt.plot(y_line_center*np.ones(len(y_line_arr)), np.arange(len(y_line_arr)))

    # ax2 = plt.subplot(212, sharex=ax1)
    # plt.plot(np.arange(len(y_line_arr)), y_line_arr)
    # plt.show()
    return y_line_center

    """
    y_line['z.count'] = y_line['z.count'].rolling(10, min_periods=1, center=True).max()
    y_line['z.count'] = y_line['z.count'].rolling(10, min_periods=1, center=True).min()

    z_count_arr = y_line['z.count'].values
    z_count_backward_shift = np.zeros(len(z_count_arr))
    z_count_backward_shift[:-1] = z_count_arr[1:] - z_count_arr[:-1]

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(y_line['y'], y_line['z.count'])

    ax3 = plt.subplot(313, sharex=ax1)
    plt.plot(y_line['y'], z_count_backward_shift)
    plt.show()
    """


def sternum_envelope_line(sternum_df, arr_shape, center_line, width=100, hu_threshold=400, output_prefix=None):
    """
    calculate the sternum envelope line
    :param sternum_df: sternum dataframe
    :param arr_shape: CT array shape
    :param center_line:
    :param width:
    :param hu_threshold:
    :param output_prefix:
    :return: y_envelope_min: left sternum envelope line
    :return: y_envelope_max: right sternum envelope line
    """
    y_limit_min = center_line - width
    y_limit_max = center_line + width

    sternum_df_group_by_z = sternum_df.groupby('z').agg({'y': ['min', 'max']})
    sternum_df_group_by_z.columns = ['%s.%s' % e for e in sternum_df_group_by_z.columns.tolist()]
    sternum_df_group_by_z.reset_index(inplace=True)
    sternum_df_group_by_z.rename(columns={'index': 'z'})

    y_envelope_min = np.ones(arr_shape[0]) * center_line
    y_envelope_max = np.ones(arr_shape[0]) * center_line

    index = sternum_df_group_by_z['z'].values

    y_envelope_min[index] = sternum_df_group_by_z['y.min'].values
    y_envelope_max[index] = sternum_df_group_by_z['y.max'].values

    y_envelope_min[y_envelope_min > center_line] = None
    y_envelope_min[y_envelope_min < y_limit_min] = None
    y_envelope_max[y_envelope_max < center_line] = None
    y_envelope_max[y_envelope_max > y_limit_max] = None

    y_envelope_min = pd.Series(y_envelope_min).rolling(20, min_periods=1, center=True).max()
    y_envelope_max = pd.Series(y_envelope_max).rolling(20, min_periods=1, center=True).min()

    plt.figure()
    plt.plot(y_envelope_min, np.arange(arr_shape[0]))
    plt.plot(y_envelope_max, np.arange(arr_shape[0]))
    plt.savefig('{}sternum_envelope.png'.format(output_prefix))
    return y_envelope_min, y_envelope_max


def sternum_cut(value_arr, sternum_envelope_line_left, sternum_envelope_line_right):
    """
    split sternum by using sternum envelope lines
    :param value_arr: source value array
    :param sternum_envelope_line_left: envelope line min
    :param sternum_envelope_line_right: envelope line max
    :return: None
    """
    z_index = np.arange(len(sternum_envelope_line_left), dtype=np.int16)
    x_index = np.ones(len(sternum_envelope_line_left), dtype=np.int16)
    y_left_index = np.array(sternum_envelope_line_left, dtype=np.int16)
    y_right_index = np.array(sternum_envelope_line_right, dtype=np.int16)

    for i in range(2*value_arr.shape[1]//3):
        value_arr[z_index, i * x_index, y_left_index] = 0
        value_arr[z_index, i * x_index, y_right_index] = 0


def final_filter(arr):
    """
    filter ribs at last
    :param arr: array just containing ribs
    :return: real ribs array
    """
    pass

