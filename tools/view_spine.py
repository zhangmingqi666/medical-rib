
import pandas as pd
import numpy as np
from preprocessing.separated.ribs_obtain.util import (plot_yzd, sparse_df_to_arr, arr_to_sparse_df, timer,
                                                      loop_morphology_binary_opening, source_hu_value_arr_to_binary,
                                                      arr_to_sparse_df_only, plot_binary_array)

apart_distance = 100


def railings_relu_interpolate(spine_remaining_df=None):
    # groupby y.statistics by z, y.std Equivalent to radius @ z
    _y_center_df = spine_remaining_df.groupby('z').agg({'y': ['mean', 'min', 'max', 'std']})
    _y_center_df.columns = ['%s.%s' % e for e in _y_center_df.columns]

    y_std_mean, y_std_std = _y_center_df['y.std'].mean(), _y_center_df['y.std'].std()
    _y_center_df['y.std.distribution'] = _y_center_df['y.std'].apply(lambda x: np.abs(x - y_std_mean) / y_std_std)
    _y_center_df[_y_center_df['y.std.distribution'] > 1.0] = np.NAN
    interpolate_columns = ['y.mean', 'y.min', 'y.max', 'y.std']
    _y_center_df[interpolate_columns] = _y_center_df[interpolate_columns].interpolate()
    _y_center_df[interpolate_columns] = _y_center_df[interpolate_columns].fillna(method='ffill').fillna(method='bfill')
    _y_center_df.reset_index(inplace=True)
    _y_center_df.rename(columns={'index': 'z'})
    _y_center_df['y.mean'] = _y_center_df['y.mean'].rolling(30).mean().fillna(method='ffill').fillna(method='bfill')

    return _y_center_df


def railings_relu_max_mean_interpolate(_center_df=None, ):
    _center_df['y.min.rolling.mean'] = _center_df['y.min'].rolling(40, min_periods=15).max().fillna(method='ffill').fillna(method='bfill')
    _center_df['y.max.rolling.mean'] = _center_df['y.max'].rolling(40, min_periods=15).min().fillna(method='ffill').fillna(method='bfill')
    _center_df['y.min.na'] = _center_df.apply(lambda row: row['y.min'] - row['y.min.rolling.mean'] < -30, axis=1)
    _center_df['y.max.na'] = _center_df.apply(lambda row: row['y.max'] - row['y.max.rolling.mean'] > 30, axis=1)
    _center_df['y.min'] = _center_df.apply(lambda row: None if row['y.min.na'] else row['y.min'], axis=1)
    _center_df['y.max'] = _center_df.apply(lambda row: None if row['y.max.na'] else row['y.max'], axis=1)
    #_center_df['y.min'] = _center_df['y.min'].interpolate().fillna(method='ffill').fillna(method='bfill')
    #_center_df['y.max'] = _center_df['y.max'].interpolate().fillna(method='ffill').fillna(method='bfill')
    #_center_df['y.min'] = _center_df.apply(lambda row: row['y.min'] - 20 if row['y.min.na'] else row['y.min'], axis=1)
    #_center_df['y.max'] = _center_df.apply(lambda row: row['y.max'] + 20 if row['y.max.na'] else row['y.max'], axis=1)
    return _center_df


ids = ['135402000404222', '135402000404891', '135402000404065', '135402000357765', '135402000555091',
       '135402000572309', '135402000555684', '135402000404090']

for id in ids:
#id = 135402000357765
    path = '../experiments/logs/{}/is_spine_opening_{}th.csv'.format(id, 1)
    spine_remaining_df = pd.read_csv(path)
    y_mean = spine_remaining_df['y'].mean()
    y_left, y_right = y_mean - apart_distance, y_mean + apart_distance
    apart_spine_remaining_df = spine_remaining_df[(spine_remaining_df['y'] > y_left) &
                                                  (spine_remaining_df['y'] < y_right)]
    y_center_df = railings_relu_interpolate(spine_remaining_df=apart_spine_remaining_df)

    temp_df = apart_spine_remaining_df.merge(y_center_df, on='z', how='inner')
    temp_df['inner'] = temp_df.apply(lambda row: np.abs(row['y']-row['y.mean'])/row['y.std'], axis=1)
    apart_spine_remaining_df_enhanced = temp_df[temp_df['inner'] < 3.0]
    y_center_df_enhanced = railings_relu_interpolate(spine_remaining_df=apart_spine_remaining_df_enhanced)

    # y_center_df_enhanced = railings_relu_max_mean_interpolate(_center_df=y_center_df_enhanced)

    z_shape = spine_remaining_df['z'].max() + 1
    y_shape = spine_remaining_df['y'].max() + 1
    print("#"*20 + id)
    # print(y_center_df_enhanced[y_center_df_enhanced['y.min.na']], y_center_df_enhanced[y_center_df_enhanced['y.max.na']])
    for e in spine_remaining_df['c'].unique():
        class_df = spine_remaining_df[spine_remaining_df['c']==e]
        class_z_mean, class_x_mean, class_y_mean = class_df['z'].mean(), class_df['x'].mean(), class_df['y'].mean()
        print("class_id : {}, center: {}".format(e, (class_z_mean, class_x_mean, class_y_mean)))
        plot_yzd(temp_df=spine_remaining_df[spine_remaining_df['c']==e], shape_arr=(z_shape, y_shape),
                 save=True, save_path='../experiments/logs/{}/remaining_spine_{}.png'.format(id, e),
                 line_tuple_list=[(y_center_df_enhanced['z'], y_center_df_enhanced['y.mean']),
                                  (y_center_df_enhanced['z'], y_center_df_enhanced['y.min']),
                                  (y_center_df_enhanced['z'], y_center_df_enhanced['y.max'])])

        # plot_yzd(temp_df=spine_remaining_df, shape_arr=(z_shape, y_shape),
        #          save=True, save_path='../experiments/merge_ids/spine_{}.png'.format(id),
        #          line_tuple_list=[(_y_center_df['z'], _y_center_df['y.mean']),
        #                           (_y_center_df['z'], _y_center_df['y.mean'] - _y_center_df['y.std']),
        #                           (_y_center_df['z'], _y_center_df['y.mean'] + _y_center_df['y.std'])])


