#coding=utf-8
import pandas as pd
import os


def dfs_read_dcm(root_file=None):

    for f in os.listdir(root_file):
        if f.endswith('.dcm'):
            return root_file
        next_dir = os.path.join(root_file, f)
        if not os.path.isdir(next_dir):
            continue
        res = dfs_read_dcm(root_file=next_dir)
        if res is not None:
            return res
    return None


def find_all_dcm_info_and_its_id_for_one_batch_data(path=None):

    # def read_info_by_one_dcm(one_dcm_path=None):
    #    slice = dicom.read_file(one_dcm_path)
    #    return [slice.SliceThickness, slice.PixelSpacing[0], slice.PixelSpacing[1]]
    id_list = []
    dcm_path_list = []
    # dcm_one_file_list = []
    z_pixel_spacing = []
    x_pixel_spacing = []
    y_pixel_spacing = []

    for f in os.listdir(path):
        next_dir = os.path.join(path, f)
        if not os.path.isdir(next_dir):
            continue

        res = dfs_read_dcm(next_dir)
        if res is not None:
            id_list.append(f)
            dcm_path_list.append(res)
            # dcm_one_file_list.append(res[1])
            # pixel_spacing = read_info_by_one_dcm(one_dcm_path=res[1])
            # z_pixel_spacing.append(pixel_spacing[0])
            # x_pixel_spacing.append(pixel_spacing[1])
            # y_pixel_spacing.append(pixel_spacing[2])

    # 'dcm_path': read all dcm to get pkl
    return pd.DataFrame({'id': id_list, 'dcm_path': dcm_path_list})


def find_all_dcm_info_and_its_id(paths):
    df = pd.DataFrame({})
    for path in paths:
        temp_df = find_all_dcm_info_and_its_id_for_one_batch_data(path)
        df = df.append(temp_df)
    return df



if __name__ == '__main__':
    pass
    # dcm_df = find_all_dcm_info_and_its_id(path='/Users/jiangyy/projects/medical-rib/dataset/dataset_mark')
    # print(dcm_df)
    # location_df = location_read(folder_path='/Users/jiangyy/Desktop/rib_fracture')
    # df = df.merge(location_df, on='id')
    # df.to_csv('hello.csv')











