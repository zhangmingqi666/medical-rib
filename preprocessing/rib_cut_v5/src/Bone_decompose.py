import numpy as np
import pandas as pd
import time
from .Bone import Bone
from .Bone_prior import BonePrior
from .Spine_Remove import SpineRemove
from .Sternum_Remove import SternumRemove
import skimage.morphology as sm
import skimage
import matplotlib.pyplot as plt
import gc
from skimage.measure import label
from .util import *
from .Bone_prior import *
import pickle as pkl
import re


def skull_remove_operation(value_arr, hu_threshold=150, bone_prior=None, allow_debug=False, output_prefix=None):
    binary_arr = value_arr.copy()
    binary_arr[binary_arr < hu_threshold] = 0
    binary_arr[binary_arr >= hu_threshold] = 1
    label_arr = skimage.measure.label(binary_arr, connectivity=2)
    del binary_arr

    sparse_df, cluster_df = arr_to_sparse_df(label_arr=label_arr, add_pixel=True, pixel_arr=value_arr,
                                             sort=True, sort_key='c.count',
                                             keep_by_top=True, top_nth=20,
                                             keep_by_threshold=True, threshold_min=5000)
    del label_arr

    for e in cluster_df['c'].values:
        temp_sparse_df = sparse_df[sparse_df['c'] == e]
        # will add center line , @issac
        single_bone = Bone(bone_data=temp_sparse_df, arr_shape=bone_prior.get_prior_shape(), spine_width=100,
                           prior_zoy_center_y_axis_line_df=bone_prior.get_zoy_symmetric_y_axis_line_df())
        if single_bone.is_skull():
            plot_yzd(temp_df=temp_sparse_df, shape_arr=(value_arr.shape[0], value_arr.shape[2]), save=True,
                     save_path='{}sternum_remaining.png'.format(output_prefix))
            skull_arr = sparse_df_to_arr(arr_expected_shape=value_arr.shape, sparse_df=temp_sparse_df, fill_bool=True)
            value_arr[skull_arr > 0] = 0
            del skull_arr
        del single_bone
        gc.collect()


def judge_collect_spine_judge_connected_rib(sparse_df=None, cluster_df=None, bone_prior=None, output_prefix=None, opening_times=None):
    """
    Combine all spine bone , and decide whether the combine spine connected ribs?
    :return loc_spine_connected_rib: if the remaining bone connect with ribs?
    :return remaining_bone_df: the remaining spine
    :return sternum_bone_df: the sternum
    """
    remaining_bone_df = pd.DataFrame({})
    sternum_bone_df = pd.DataFrame({})
    loc_spine_connected_rib = False
    for e in cluster_df['c'].values:
        temp_sparse_df = sparse_df[sparse_df['c'] == e]
        # will add center line , @issac
        single_bone = Bone(bone_data=temp_sparse_df, arr_shape=bone_prior.get_prior_shape(), spine_width=100,
                           prior_zoy_center_y_axis_line_df=bone_prior.get_zoy_symmetric_y_axis_line_df(),
                           detection_objective='spine or sternum')
        single_bone.set_bone_type()
        plt.figure()
        z_max = single_bone.local_max_for_sternum[0]
        x_center = single_bone.local_centroid_for_sternum[1]
        plt.title('os:%d, r:%s, p_cnt:%s, y_mx_on_z:%d, spine:%d' % (opening_times, cluster_df[cluster_df['c'] == e].index[0], len(temp_sparse_df),
                                                                     single_bone.get_y_length_statistics_on_z(feature='max'), single_bone.is_spine()), color='red')
        single_bone.plot_bone(save=True, save_path='{}opening_{}_label_{}_collect.png'.format(output_prefix, opening_times, e))

        # single_bone.plot_bone()
        if single_bone.is_spine():
            # single_bone.plot_bone()
            remaining_bone_df = remaining_bone_df.append(single_bone.get_bone_data())
            if single_bone.spine_connected_rib():
                if int(len(temp_sparse_df)) == 168877:
                    print("肋骨判断失误！！！")
                    print('%d > %d' % (single_bone.get_y_length_statistics_on_z(feature='max'), single_bone.spine_connected_rib_y_length_thresholds))
                loc_spine_connected_rib = True

        if single_bone.is_sternum():
            # single_bone.plot_bone()
            sternum_bone_df = sternum_bone_df.append(single_bone.get_bone_data())
        del single_bone
    # print(sternum_bone_df)
    return loc_spine_connected_rib, remaining_bone_df, sternum_bone_df


def collect_ribs(value_arr, hu_threshold=150, bone_prior=None, allow_debug=False, output_prefix=None):
    """
    combine all ribs from source CT array
    :param value_arr:
    :param hu_threshold:
    :param bone_prior:
    :param allow_debug:
    :param output_prefix:
    :return: rib_bone_df: containing all ribs' index(z,x,y)
    """
    patient_id = output_prefix.split("/")[-2]
    binary_arr = value_arr.copy()
    binary_arr[binary_arr < hu_threshold] = 0
    binary_arr[binary_arr >= hu_threshold] = 1
    label_arr = skimage.measure.label(binary_arr, connectivity=1)
    del binary_arr

    sparse_df, cluster_df = arr_to_sparse_df(label_arr=label_arr, add_pixel=True, pixel_arr=value_arr,
                                             sort=True, sort_key='c.count',
                                             keep_by_top=True, top_nth=40,
                                             keep_by_threshold=True, threshold_min=5000)
    del label_arr

    rib_bone_df = pd.DataFrame({})
    for e in cluster_df['c'].values:
        temp_sparse_df = sparse_df[sparse_df['c'] == e]
        # will add center line , @issac
        single_bone = Bone(bone_data=temp_sparse_df, arr_shape=bone_prior.get_prior_shape(), spine_width=100,
                           prior_zoy_center_y_axis_line_df=bone_prior.get_zoy_symmetric_y_axis_line_df())
        single_bone.set_bone_type()
        print('|||||||||||||||||||||||||||||||||||||||||||||||')
        print('num:%d' % e)
        single_bone.print_bone_info()
        print('|||||||||||||||||||||||||||||||||||||||||||||||')
        single_bone.plot_bone(save=True, save_path='{}label_{}_collect_{}.png'.format(output_prefix, e, single_bone.rib_type))

        #     pkl.dump(single_bone.get_bone_data(), open('/home/jiangyy/Desktop/%d.pkl' % e, 'wb'))
        if single_bone.is_rib():
            rib_bone_df = rib_bone_df.append(single_bone.get_bone_data())
            df_file_path = '/home/jiangyy/Desktop/rib_df_input/%s/%d.csv' % (patient_id, e)
            # pkl.dump(single_bone.get_bone_data()[['x', 'y', 'z', 'v']], open(df_file_path, 'wb'))
            # single_bone.get_bone_data()[['x', 'y', 'z', 'v']].to_csv(df_file_path)
        del single_bone
    return rib_bone_df


def loop_opening_get_spine(binary_arr, hu_threshold=400, bone_prior=None, allow_debug=False, output_prefix=None):

    # calc bone prior
    sternum_bone_df = pd.DataFrame({})
    opening_times = 0
    while True:
        # circulation tiems

        with timer('_________label'):
            label_arr = skimage.measure.label(binary_arr, connectivity=2)
        del binary_arr

        with timer('_________arr to sparse df'):
            # add select objective.
            sparse_df, cluster_df = arr_to_sparse_df(label_arr=label_arr, sort=True, sort_key='c.count',
                                                     keep_by_top=True, top_nth=100,
                                                     keep_by_threshold=True, threshold_min=1000)
        del label_arr
        with timer('_________collect spine and judge connected'):
            glb_spine_connected_rib, remaining_bone_df, temp = judge_collect_spine_judge_connected_rib(sparse_df=sparse_df,
                                                                                                       cluster_df=cluster_df,
                                                                                                       bone_prior=bone_prior,
                                                                                                       output_prefix=output_prefix,
                                                                                                       opening_times=opening_times)
        if temp is not None:
            sternum_bone_df = sternum_bone_df.append(temp)
        del sparse_df, cluster_df, temp

        if glb_spine_connected_rib is False:
            break
        else:
            opening_times = opening_times + 1

        with timer('_________sparse df to arr'):
            binary_arr = sparse_df_to_arr(arr_expected_shape=bone_prior.get_prior_shape(), sparse_df=remaining_bone_df)
            binary_arr[binary_arr > 0] = 1
        del remaining_bone_df

        with timer('_________binary opening'):
            binary_arr = loop_morphology_binary_opening(binary_arr, use_cv=False)

    return remaining_bone_df, sternum_bone_df


def main_excute(value_arr, allow_debug=False, output_prefix='hello'):
    with timer('morphology value arr to binary'):
        binary_arr = morphology_value_arr_to_binary(value_arr=value_arr)

    with timer('split sternum form circle ribs'):
        # sternum_center_line = calc_sternum_center_line(value_arr, hu_threshold=400)
        half_front_bone_df = arr_to_sparse_df_only(binary_arr=binary_arr[:, :binary_arr.shape[1]//2, :])
        left_envelope_line, right_envelope_line = sternum_envelope_line(sternum_df=half_front_bone_df, arr_shape=value_arr.shape, width=100,
                                                                        center_line=value_arr.shape[2]//2, output_prefix=output_prefix)
        # sternum_cut(value_arr, left_envelope_line, right_envelope_line)
        plt.figure()
        plt.title('half_bone')
        plt.imshow(binary_arr[:, :binary_arr.shape[1]//2, :].sum(axis=1))
        plt.plot(left_envelope_line, np.arange(value_arr.shape[0]))
        plt.plot(right_envelope_line, np.arange(value_arr.shape[0]))
        plt.savefig('{}half_front_bones_with_envelope_line.png'.format(output_prefix))

    sternum_cut(value_arr=value_arr, sternum_envelope_line_left=left_envelope_line, sternum_envelope_line_right=right_envelope_line)

    with timer('calc bone prior'):
        bone_prior = BonePrior(binary_arr=binary_arr)
    """
    plt.figure()
    plt.title('half_bone')
    plt.imshow(binary_arr[:, :binary_arr.shape[1]//2, :].sum(axis=1))
    plt.savefig('{}half_front_bones.png'.format(output_prefix))
    """
    """
    with timer('remove the skull'):
        skull_remove_operation(value_arr=value_arr, hu_threshold=150, bone_prior=bone_prior)
    """
    if False:
        zy = bone_prior.get_zoy_symmetric_y_axis_line_df()
        plt.plot(zy['y.center'], zy['z'], c='r')
        plt.plot(zy['filter_mean'], zy['z'], c='y')
        plt.imshow(binary_value_arr.sum(axis=1))
        plt.show()

    with timer('get spine remaining df'):
        remaining_bone_df, sternum_bone_df = loop_opening_get_spine(binary_arr, hu_threshold=400,
                                                                    bone_prior=bone_prior, allow_debug=allow_debug, output_prefix=output_prefix)

        shape = bone_prior.get_prior_shape()
        if len(remaining_bone_df) == 0:
            print("脊柱为空！！！")
        else:
            plot_yzd(temp_df=remaining_bone_df, shape_arr=(shape[0], shape[2]), save=True, save_path='{}spine_remaining.png'.format(output_prefix))
        if len(sternum_bone_df) == 0:
            print("胸骨为空！！！")
        else:
            plot_yzd(temp_df=sternum_bone_df, shape_arr=(shape[0], shape[2]), save=True, save_path='{}sternum_remaining.png'.format(output_prefix))


    # first split the sternum from circle ribs


    if False:
        temp_debug_arr = np.zeros(bone_prior.get_prior_shape())
        temp_debug_index = remaining_bone_df['z'].values, remaining_bone_df['x'].values, remaining_bone_df['y'].values
        temp_debug_arr[temp_debug_index] = 1
        plt.imshow(temp_debug_arr.sum(axis=1))
        plt.show()
        del temp_debug_arr, temp_debug_index
        gc.collect()

    with timer('spine remove calculation'):
        # shape = bone_prior.get_prior_shape()
        # if len(remaining_bone_df) != 0:
        # plot_yzd(temp_df=remaining_bone_df, shape_arr=(shape[0], shape[2]), save=True, save_path='{}spine_remaining.png'.format(output_prefix))
        spine_remove = SpineRemove(bone_data_df=remaining_bone_df, bone_data_shape=bone_prior.get_prior_shape(),
                                   allow_envelope_expand=True, expand_width=20)
        # version 3.1.3, speed up from 120s to 0s
        spine_remove.spine_remove_operation(value_arr=value_arr)
    """
    with timer('sternum remove calculation'):
        if len(sternum_bone_df) != 0:
            print(sternum_bone_df)
            plot_yzd(temp_df=sternum_bone_df, shape_arr=(shape[0], shape[2]), save=True,
                     save_path='{}sternum_remaining.png'.format(output_prefix))
            sternum_remove = SternumRemove(bone_data_df=sternum_bone_df, allow_envelope_expand=True, expand_width=20)
            sternum_remove.sternum_remove_operation(value_arr=value_arr)
            # version 3.1.3, speed up from 120s to 0s
            # sternum_remove.sternum_remove_operation(value_arr=value_arr)
    """
    with timer('collect ribs'):
        rib_bone_df = collect_ribs(value_arr, hu_threshold=400, bone_prior=bone_prior, output_prefix=output_prefix)

    if allow_debug:
        restore_arr = np.zeros(bone_prior.get_prior_shape())
        if len(rib_bone_df) > 0:
            rib_index_all = rib_bone_df['z'].values, rib_bone_df['x'].values, rib_bone_df['y'].values
            restore_arr[rib_index_all] = 1
            plt.imshow(restore_arr.sum(axis=1))
            plt.savefig('{}collect_ribs.png'.format(output_prefix))
        else:
            print("rib bone df = 0")
        # plt.show()
