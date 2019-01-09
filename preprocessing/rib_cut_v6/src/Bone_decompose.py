import numpy as np
import pandas as pd
import time
import skimage.morphology as sm
import skimage
import matplotlib.pyplot as plt
import gc
import pickle as pkl
from skimage.measure import label
from .util import *
from .Bone_prior import *
from .Bone import Bone
from .Bone_prior import BonePrior
from .Spine_Remove import SpineRemove
from .Remove_Sternum import SternumRemove
from sklearn.externals import joblib
# from .Sternum_Remove import SternumRemove


def skull_remove_operation(value_arr, hu_threshold=150, bone_prior=None, allow_debug=False, output_prefix=None):
    """
    remove skull bone
    :param value_arr:
    :param hu_threshold:
    :param bone_prior:
    :param allow_debug:
    :param output_prefix:
    :return: None
    """
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

        # save single bone fig
        plt.figure()
        # z_max = single_bone.local_max_for_sternum[0]
        # x_center = single_bone.local_centroid_for_sternum[1]
        plt.title('os:%d, r:%s, p_cnt:%s, y_mx_on_z:%d, mean_z_on_x:%s' % (opening_times, cluster_df[cluster_df['c'] == e].index[0],
                                                                           len(temp_sparse_df), single_bone.get_y_length_statistics_on_z(feature='max'),
                                                                           single_bone.get_mean_z_distance_on_xoy()), color='red')
        single_bone.plot_bone(save=True, save_path='{}opening_{}_label_{}_collect.png'.format(output_prefix, opening_times, e))

        if single_bone.is_spine():
            remaining_bone_df = remaining_bone_df.append(single_bone.get_bone_data())
            if single_bone.spine_connected_rib():
                loc_spine_connected_rib = True

        if single_bone.is_sternum():
            sternum_bone_df = sternum_bone_df.append(single_bone.get_bone_data())
        del single_bone

    return loc_spine_connected_rib, remaining_bone_df, sternum_bone_df


def collect_ribs(value_arr, hu_threshold=150, bone_prior=None, allow_debug=False, output_prefix=None):
    """
    combine all ribs from source CT array after removing spine and sternum
    :param value_arr: HU array after removing spine and sternum
    :param hu_threshold: 150
    :param bone_prior:
    :param allow_debug:
    :param output_prefix:
    :return: rib_bone_df: containing all ribs' index(z,x,y)
    """
    # generate binary array from source HU array for labeling
    binary_arr = value_arr.copy()
    binary_arr[binary_arr < hu_threshold] = 0
    binary_arr[binary_arr >= hu_threshold] = 1
    # bone labeling
    label_arr = skimage.measure.label(binary_arr, connectivity=2)
    del binary_arr

    sparse_df, cluster_df = arr_to_sparse_df(label_arr=label_arr, add_pixel=True, pixel_arr=value_arr,
                                             sort=True, sort_key='c.count',
                                             keep_by_top=True, top_nth=40,
                                             keep_by_threshold=True, threshold_min=5000)
    del label_arr

    rib_bone_df = pd.DataFrame({})
    bone_info_df = pd.DataFrame({}, columns=['label', 'z_centroid/z_shape', 'x_centroid/x_shape', 'y_centroid/y_shape', 'z_max/z_shape',
                                             'x_max/x_shape', 'y_max/y_shape', 'z_min/z_shape', 'x_min/x_shape', 'y_min/y_shape', 'z_length/z_shape',
                                             'x_length/x_shape', 'y_length/y_shape', 'iou_on_xoy', 'distance_nearest_centroid', 'point_count',
                                             'mean_z_distance_on_xoy', 'max_nonzero_internal'])

    patient_id = output_prefix.split("/")[-2]

    # load gbdt model
    gbdt = joblib.load('/home/jiangyy/projects/medical-rib/model/gbdt.pkl')
    for e in cluster_df['c'].values:
        temp_sparse_df = sparse_df[sparse_df['c'] == e]
        # will add center line , @issac
        single_bone = Bone(bone_data=temp_sparse_df, arr_shape=bone_prior.get_prior_shape(), spine_width=100,
                           prior_zoy_center_y_axis_line_df=bone_prior.get_zoy_symmetric_y_axis_line_df())
        single_bone.set_bone_type()
        print('***********************************************')
        print('bone class num:%d' % e)
        single_bone.print_bone_info()
        print('***********************************************')
        single_bone.plot_bone(save=True, save_path='{}label_{}_collect_{}.png'.format(output_prefix, e, single_bone.rib_type))

        temp_single_bone_df = pd.DataFrame({'label': [e],
                                            'z_centroid/z_shape': [single_bone.get_basic_axis_feature(feature='centroid')[0] / single_bone.arr_shape[0]],
                                            'x_centroid/x_shape': [single_bone.get_basic_axis_feature(feature='centroid')[1] / single_bone.arr_shape[1]],
                                            'y_centroid/y_shape': [single_bone.get_basic_axis_feature(feature='centroid')[2] / single_bone.arr_shape[2]],
                                            'z_max/z_shape': [single_bone.get_basic_axis_feature(feature='max')[0] / single_bone.arr_shape[0]],
                                            'x_max/x_shape': [single_bone.get_basic_axis_feature(feature='max')[1] / single_bone.arr_shape[1]],
                                            'y_max/y_shape': [single_bone.get_basic_axis_feature(feature='max')[2] / single_bone.arr_shape[2]],
                                            'z_min/z_shape': [single_bone.get_basic_axis_feature(feature='min')[0] / single_bone.arr_shape[0]],
                                            'x_min/x_shape': [single_bone.get_basic_axis_feature(feature='min')[1] / single_bone.arr_shape[1]],
                                            'y_min/y_shape': [single_bone.get_basic_axis_feature(feature='min')[2] / single_bone.arr_shape[2]],
                                            'z_length/z_shape': [single_bone.get_basic_axis_feature(feature='length')[0] / single_bone.arr_shape[0]],
                                            'x_length/x_shape': [single_bone.get_basic_axis_feature(feature='length')[1] / single_bone.arr_shape[1]],
                                            'y_length/y_shape': [single_bone.get_basic_axis_feature(feature='length')[2] / single_bone.arr_shape[2]],
                                            'iou_on_xoy': [single_bone.get_iou_on_xoy()],
                                            'distance_nearest_centroid': [single_bone.get_distance_between_centroid_and_nearest_point()],
                                            'point_count': [len(single_bone.get_bone_data())],
                                            'mean_z_distance_on_xoy': [single_bone.get_mean_z_distance_on_xoy()],
                                            'max_nonzero_internal': [single_bone.detect_multi_ribs()],
                                            'is_rib': [1 if single_bone.is_rib() else 0]
                                            })
        # dump csv file

        # temp_single_bone_df['label'][0] = e
        # temp_single_bone_df['z_centroid/z_shape'][0] = single_bone.get_basic_axis_feature(feature='centroid')[0] / single_bone.arr_shape[0]
        # temp_single_bone_df['x_centroid/x_shape'][0] = single_bone.get_basic_axis_feature(feature='centroid')[1] / single_bone.arr_shape[1]
        # temp_single_bone_df['y_centroid/y_shape'][0] = single_bone.get_basic_axis_feature(feature='centroid')[2] / single_bone.arr_shape[2]
        # temp_single_bone_df['z_max/z_shape'][0] = single_bone.get_basic_axis_feature(feature='max')[0] / single_bone.arr_shape[0]
        # temp_single_bone_df['x_max/x_shape'][0] = single_bone.get_basic_axis_feature(feature='max')[1] / single_bone.arr_shape[1]
        # temp_single_bone_df['y_max/y_shape'][0] = single_bone.get_basic_axis_feature(feature='max')[2] / single_bone.arr_shape[2]
        # temp_single_bone_df['z_min/z_shape'][0] = single_bone.get_basic_axis_feature(feature='min')[0] / single_bone.arr_shape[0]
        # temp_single_bone_df['x_min/x_shape'][0] = single_bone.get_basic_axis_feature(feature='min')[1] / single_bone.arr_shape[1]
        # temp_single_bone_df['y_min/y_shape'][0] = single_bone.get_basic_axis_feature(feature='min')[2] / single_bone.arr_shape[2]
        # temp_single_bone_df['z_length/z_shape'][0] = single_bone.get_basic_axis_feature(feature='length')[0] / single_bone.arr_shape[0]
        # temp_single_bone_df['x_length/x_shape'][0] = single_bone.get_basic_axis_feature(feature='length')[1] / single_bone.arr_shape[1]
        # temp_single_bone_df['y_length/y_shape'][0] = single_bone.get_basic_axis_feature(feature='length')[2] / single_bone.arr_shape[2]
        # temp_single_bone_df['iou_on_xoy'][0] = single_bone.get_iou_on_xoy()
        # temp_single_bone_df['distance_nearest_centroid'][0] = single_bone.get_distance_between_centroid_and_nearest_point()
        # temp_single_bone_df['point_count'][0] = len(single_bone.get_bone_data())
        # temp_single_bone_df['mean_z_distance_on_xoy'][0] = single_bone.get_mean_z_distance_on_xoy()
        # temp_single_bone_df['max_nonzero_internal'][0] = single_bone.detect_multi_ribs()

        bone_info_df = bone_info_df.append(temp_single_bone_df)

        # else:
        #     bone_info_df['label'] = bone_info_df['label'].append(e)
        #     bone_info_df['z_centroid/z_shape'] = bone_info_df['z_centroid/z_shape'].append(single_bone.get_basic_axis_feature(feature='centroid')[0] / single_bone.arr_shape[0])
        #     bone_info_df['x_centroid/x_shape'] = bone_info_df['x_centroid/x_shape'].append(single_bone.get_basic_axis_feature(feature='centroid')[1] / single_bone.arr_shape[1])
        #     bone_info_df['y_centroid/y_shape'] = bone_info_df['y_centroid/y_shape'].append(single_bone.get_basic_axis_feature(feature='centroid')[2] / single_bone.arr_shape[2])
        #     bone_info_df['z_max/z_shape'] = bone_info_df['z_max/z_shape'].append(single_bone.get_basic_axis_feature(feature='max')[0] / single_bone.arr_shape[0])
        #     bone_info_df['x_max/x_shape'] = bone_info_df['x_max/x_shape'].append(single_bone.get_basic_axis_feature(feature='max')[1] / single_bone.arr_shape[1])
        #     bone_info_df['y_max/y_shape'] = bone_info_df['y_max/y_shape'].append(single_bone.get_basic_axis_feature(feature='max')[2] / single_bone.arr_shape[2])
        #     bone_info_df['z_min/z_shape'] = bone_info_df['z_min/z_shape'].append(single_bone.get_basic_axis_feature(feature='min')[0] / single_bone.arr_shape[0])
        #     bone_info_df['x_min/x_shape'] = bone_info_df['x_min/x_shape'].append(single_bone.get_basic_axis_feature(feature='min')[1] / single_bone.arr_shape[1])
        #     bone_info_df['y_min/y_shape'] = bone_info_df['y_min/y_shape'].append(single_bone.get_basic_axis_feature(feature='min')[2] / single_bone.arr_shape[2])
        #     bone_info_df['z_length/z_shape'] = bone_info_df['z_length/z_shape'].append(single_bone.get_basic_axis_feature(feature='length')[0] / single_bone.arr_shape[0])
        #     bone_info_df['x_length/x_shape'] = bone_info_df['x_length/x_shape'].append(single_bone.get_basic_axis_feature(feature='length')[1] / single_bone.arr_shape[1])
        #     bone_info_df['y_length/y_shape'] = bone_info_df['y_length/y_shape'].append(single_bone.get_basic_axis_feature(feature='length')[2] / single_bone.arr_shape[2])
        #     bone_info_df['iou_on_xoy'] = bone_info_df['iou_on_xoy'].append(single_bone.get_iou_on_xoy())
        #     bone_info_df['distance_nearest_centroid'] = bone_info_df['iou_on_xoy'].append(single_bone.get_distance_between_centroid_and_nearest_point())
        #     bone_info_df['point_count'] = bone_info_df['point_count'].append(len(single_bone.get_bone_data()))
        #     bone_info_df['mean_z_distance_on_xoy'] = bone_info_df['mean_z_distance_on_xoy'].append(single_bone.get_mean_z_distance_on_xoy())
        #     bone_info_df['max_nonzero_internal'] = bone_info_df['max_nonzero_internal'].append(single_bone.detect_multi_ribs())

        # if single_bone.is_rib():
        print("gbdt", gbdt.predict(temp_single_bone_df[['z_centroid/z_shape', 'x_centroid/x_shape', 'y_centroid/y_shape', 'z_max/z_shape', 'x_max/x_shape',
                                                        'y_max/y_shape', 'z_min/z_shape', 'x_min/x_shape', 'y_min/y_shape', 'z_length/z_shape', 'x_length/x_shape',
                                                        'y_length/y_shape', 'iou_on_xoy', 'distance_nearest_centroid', 'point_count', 'mean_z_distance_on_xoy', 'max_nonzero_internal']]))
        if gbdt.predict(temp_single_bone_df[['z_centroid/z_shape', 'x_centroid/x_shape', 'y_centroid/y_shape', 'z_max/z_shape', 'x_max/x_shape',
                                             'y_max/y_shape', 'z_min/z_shape', 'x_min/x_shape', 'y_min/y_shape', 'z_length/z_shape', 'x_length/x_shape',
                                             'y_length/y_shape', 'iou_on_xoy', 'distance_nearest_centroid', 'point_count', 'mean_z_distance_on_xoy', 'max_nonzero_internal']]):
            rib_bone_df = rib_bone_df.append(single_bone.get_bone_data())
            df_file_path = '/home/jiangyy/Desktop/rib_df_input/%s/%d.csv' % (patient_id, e)
            # pkl.dump(single_bone.get_bone_data()[['x', 'y', 'z', 'v']], open(df_file_path, 'wb'))
            # save single rib dataframe to csv file
            # single_bone.get_bone_data()[['x', 'y', 'z', 'v']].to_csv(df_file_path)
        del single_bone

    bone_info_df.sort_values(by='label', inplace=True)
    # bone_info_df.to_csv('/home/jiangyy/projects/medical-rib/bone_df_csv/%s.csv' % patient_id)
    return rib_bone_df


def loop_opening_get_spine(binary_arr, hu_threshold=400, bone_prior=None, allow_debug=False, output_prefix=None):

    # calc bone prior
    sternum_bone_df = pd.DataFrame({})
    opening_times = 0
    while True:
        # circulation times

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
            sternum_bSternumRemoveone_df = sternum_bone_df.append(temp)
        del sparse_df, cluster_df, temp

        if (glb_spine_connected_rib is False) or (opening_times > 4):
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


def plot_binary_array(binary_arr, title=None, save=True, fig_name=None, output_prefix=None):
    plt.figure()
    plt.title(title, color='red')
    plt.imshow(binary_arr.sum(axis=1))
    if save:
        plt.savefig('{}{}.png'.format(output_prefix, fig_name))
    else:
        pass


def void_cut_ribs_process(value_arr, allow_debug=False, output_prefix='hello'):

    with timer('calculate basic array and feature, data frame'):
        """covert source HU array to binary array with HU threshold = 400
        """
        binary_arr = source_hu_value_arr_to_binary(value_arr=value_arr, hu_threshold=400)

        """calculate center of source CT array
        """
        # z_center = binary_arr.shape[0] // 2
        x_center = binary_arr.shape[1] // 2
        y_center = binary_arr.shape[2] // 2

    with timer('calc bone prior'):
        """calculate bone prior
        """
        bone_prior = BonePrior(binary_arr=binary_arr)

    with timer('remove sternum with sternum envelope'):
        """calculate envelope of sternum
        """
        half_front_bone_df = arr_to_sparse_df_only(binary_arr=binary_arr[:, :x_center, :])

        """remove sternum
        """
        sternum_remove = SternumRemove(bone_data_arr=value_arr, bone_data_df=half_front_bone_df,
                                       bone_data_shape=value_arr.shape, center_line=y_center,
                                       hu_threshold=400, width=100, output_prefix=output_prefix)

        sternum_remove.sternum_remove_operation(value_arr=value_arr)

        del half_front_bone_df
        gc.collect()

    with timer('get spine with looping opening'):
        """looping spine to get opening
        """
        spine_df, sternum_df = loop_opening_get_spine(binary_arr=binary_arr, hu_threshold=400, bone_prior=bone_prior,
                                                      allow_debug=allow_debug, output_prefix=output_prefix)

    with timer('remove spine from source value arr'):
        """removing spine with operating value arr
        """
        print("helologjdigdoigjdijg")
        print(bone_prior.get_prior_shape())
        spine_remove = SpineRemove(bone_data_df=spine_df, bone_data_shape=bone_prior.get_prior_shape(),
                                   allow_envelope_expand=True, expand_width=20)
        spine_remove.spine_remove_operation(value_arr=value_arr)

        del spine_remove
        gc.collect()

    with timer('collect ribs'):
        """collecting ribs from value array after removing spine and sternum
        """
        rib_bone_df = collect_ribs(value_arr, hu_threshold=150, bone_prior=bone_prior, output_prefix=output_prefix)

    """plot half front bone
    """
    if allow_debug:
        plot_binary_array(binary_arr=binary_arr[:, :x_center, :], title='half_front_bone',
                          save=False, fig_name='half_front_bone', output_prefix=output_prefix)
        plt.plot(sternum_remove.left_envelope_line, np.arange(binary_arr.shape[0]))
        plt.plot(sternum_remove.right_envelope_line, np.arange(binary_arr.shape[0]))
        plt.savefig('{}half_front_bones_with_envelope_line.png'.format(output_prefix))

    """plot split spine
    """
    if allow_debug:
        if len(spine_df) > 0:
            plot_yzd(temp_df=spine_df, shape_arr=(binary_arr.shape[0], binary_arr.shape[2]),
                     save=True, save_path='{}spine_remaining.png'.format(output_prefix))
        else:
            print("spine_df is empty!")

        if len(sternum_df) > 0:
            plot_yzd(temp_df=sternum_df, shape_arr=(binary_arr.shape[0], binary_arr.shape[2]),
                     save=True, save_path='{}sternum_remaining.png'.format(output_prefix))
        else:
            print("sternum_df is empty")

    """plot collected ribs
    """
    if allow_debug:
        restore_arr = np.zeros(bone_prior.get_prior_shape())
        if len(rib_bone_df) > 0:
            rib_index_all = rib_bone_df['z'].values, rib_bone_df['x'].values, rib_bone_df['y'].values
            restore_arr[rib_index_all] = 1
            plot_binary_array(binary_arr=restore_arr, title='collect_ribs',
                              save=True, fig_name='collect_ribs', output_prefix=output_prefix)
        else:
            print("rib bone df = 0")
        del restore_arr

    del sternum_remove
    del spine_df
    del binary_arr
    del rib_bone_df
    del bone_prior
    del value_arr
    gc.collect()
