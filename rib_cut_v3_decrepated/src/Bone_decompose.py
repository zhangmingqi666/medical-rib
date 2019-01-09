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


def judge_collect_spine_judge_connected_rib(sparse_df=None, cluster_df=None, bone_prior=None):
    """
    Combine all spine bone , and decide whether the combine spine connected ribs?
    """
    remaining_bone_df = pd.DataFrame({})
    sternum_bone_df = None
    loc_spine_connected_rib = False
    for e in cluster_df['c'].values:
        temp_sparse_df = sparse_df[sparse_df['c'] == e]
        # will add center line , @issac
        single_bone = Bone(bone_data=temp_sparse_df, arr_shape=bone_prior.get_prior_shape(), spine_width=100,
                           prior_zoy_center_y_axis_line_df=bone_prior.get_zoy_symmetric_y_axis_line_df(),
                           detection_objective='spine or sternum')
        single_bone.set_bone_type()
        # print(single_bone.is_sternum())
        # single_bone.plot_bone()
        if single_bone.is_spine():
            # single_bone.plot_bone()
            remaining_bone_df = remaining_bone_df.append(single_bone.get_bone_data())
            if single_bone.spine_connected_rib():
                loc_spine_connected_rib = True
        elif single_bone.is_sternum():
            # single_bone.plot_bone()
            sternum_bone_df = single_bone.get_bone_data()
        del single_bone
    # print(sternum_bone_df)
    return loc_spine_connected_rib, remaining_bone_df, sternum_bone_df


def collect_ribs(value_arr, hu_threshold=150, bone_prior=None, allow_debug=False, output_prefix=None):
    binary_arr = value_arr.copy()
    binary_arr[binary_arr < hu_threshold] = 0
    binary_arr[binary_arr >= hu_threshold] = 1
    label_arr = skimage.measure.label(binary_arr, connectivity=2)
    del binary_arr

    sparse_df, cluster_df = arr_to_sparse_df(label_arr=label_arr, add_pixel=True, pixel_arr=value_arr,
                                             sort=True, sort_key='c.count',
                                             keep_by_top=True, top_nth=40,
                                             keep_by_threshold=True, threshold_min=10000)
    del label_arr

    rib_bone_df = pd.DataFrame({})
    for e in cluster_df['c'].values:
        temp_sparse_df = sparse_df[sparse_df['c'] == e]
        # will add center line , @issac
        single_bone = Bone(bone_data=temp_sparse_df, arr_shape=bone_prior.get_prior_shape(), spine_width=100,
                           prior_zoy_center_y_axis_line_df=bone_prior.get_zoy_symmetric_y_axis_line_df())
        single_bone.set_bone_type()
        single_bone.print_bone_info()
        single_bone.plot_bone(save=True, save_path='_{}label_{}_collect.png'.format(output_prefix, e))
        if single_bone.is_rib():
            rib_bone_df = rib_bone_df.append(single_bone.get_bone_data())
        del single_bone
    return rib_bone_df


def loop_opening_get_spine(binary_arr, hu_threshold=400, bone_prior=None, allow_debug=False):

    # calc bone prior
    sternum_bone_df = None
    while True:

        with timer('_________label'):
            label_arr = skimage.measure.label(binary_arr, connectivity=2)
        del binary_arr

        with timer('_________arr to sparse df'):
            # add select objective.
            sparse_df, cluster_df = arr_to_sparse_df(label_arr=label_arr, sort=True, sort_key='c.count',
                                                     keep_by_top=True, top_nth=20,
                                                     keep_by_threshold=True, threshold_min=4500)
        del label_arr
        with timer('_________collect spine and judge connected'):
            glb_spine_connected_rib, remaining_bone_df, temp = judge_collect_spine_judge_connected_rib(sparse_df=sparse_df,
                                                                                                       cluster_df=cluster_df,
                                                                                                       bone_prior=bone_prior)
        if sternum_bone_df is None and temp is not None:
            sternum_bone_df = temp
        del sparse_df, cluster_df, temp

        if glb_spine_connected_rib is False:
            break

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
    with timer('calc bone prior'):
        bone_prior = BonePrior(binary_arr=binary_arr)

    if False:
        zy = bone_prior.get_zoy_symmetric_y_axis_line_df()
        plt.plot(zy['y.center'], zy['z'], c='r')
        plt.plot(zy['filter_mean'], zy['z'], c='y')
        plt.imshow(binary_value_arr.sum(axis=1))
        plt.show()

    with timer('get spine remaining df'):
        remaining_bone_df, sternum_bone_df = loop_opening_get_spine(binary_arr, hu_threshold=400,
                                                                    bone_prior=bone_prior, allow_debug=allow_debug)

    if False:
        temp_debug_arr = np.zeros(bone_prior.get_prior_shape())
        temp_debug_index = remaining_bone_df['z'].values, remaining_bone_df['x'].values, remaining_bone_df['y'].values
        temp_debug_arr[temp_debug_index] = 1
        plt.imshow(temp_debug_arr.sum(axis=1))
        plt.show()
        del temp_debug_arr, temp_debug_index
        gc.collect()

    with timer('spine remove calculation'):
        shape = bone_prior.get_prior_shape()
        plot_yzd(temp_df=remaining_bone_df, shape_arr=(shape[0], shape[2]), save=True, save_path='{}spine_remaining.png'.format(output_prefix))
        spine_remove = SpineRemove(bone_data_df=remaining_bone_df, bone_data_shape=bone_prior.get_prior_shape(),
                                   allow_envelope_expand=True, expand_width=20)
        # version 3.1.3, speed up from 120s to 0s
        spine_remove.spine_remove_operation(value_arr=value_arr)

    with timer('sternum remove calculation'):
        if sternum_bone_df is not None:
            plot_yzd(temp_df=sternum_bone_df, shape_arr=(shape[0], shape[2]), save=True,
                     save_path='{}sternum_remaining.png'.format(output_prefix))
            sternum_remove = SternumRemove(bone_data_df=sternum_bone_df, allow_envelope_expand=True, expand_width=20)
            sternum_remove.cut_sternum_v2()
            # version 3.1.3, speed up from 120s to 0s
            sternum_remove.sternum_remove_operation(value_arr=value_arr)

    with timer('collect ribs'):
        rib_bone_df = collect_ribs(value_arr, hu_threshold=150, bone_prior=bone_prior, output_prefix=output_prefix)

    if allow_debug:
        restore_arr = np.zeros(bone_prior.get_prior_shape())
        rib_index_all = rib_bone_df['z'].values, rib_bone_df['x'].values, rib_bone_df['y'].values
        restore_arr[rib_index_all] = 1
        plt.imshow(restore_arr.sum(axis=1))
        plt.savefig('{}collect_ribs.png'.format(output_prefix))
        #plt.show()





