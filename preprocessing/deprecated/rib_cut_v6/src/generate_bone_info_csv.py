import os
import pandas as pd

file_path = '/home/jiangyy/projects/medical-rib/bone_df_csv'
files = os.listdir(file_path)
all_bone_info_df = pd.DataFrame({}, columns=['label', 'z_centroid/z_shape', 'x_centroid/x_shape', 'y_centroid/y_shape', 'z_max/z_shape',
                                             'x_max/x_shape', 'y_max/y_shape', 'z_min/z_shape', 'x_min/x_shape', 'y_min/y_shape',
                                             'z_length/z_shape', 'x_length/x_shape', 'y_length/y_shape', 'iou_on_xoy', 'is_rib',
                                             'distance_nearest_centroid', 'point_count', 'mean_z_distance_on_xoy', 'max_nonzero_internal'])
for f in files:
    path = os.path.join(file_path, f)
    single_patient_bone = open(path, encoding='utf-8')
    temp_df = pd.read_csv(single_patient_bone)
    all_bone_info_df = all_bone_info_df.append(temp_df)

all_bone_info_df.reset_index(inplace=True)
all_bone_info_df = all_bone_info_df[['label', 'z_centroid/z_shape', 'x_centroid/x_shape', 'y_centroid/y_shape', 'z_max/z_shape',
                                     'x_max/x_shape', 'y_max/y_shape', 'z_min/z_shape', 'x_min/x_shape', 'y_min/y_shape',
                                     'z_length/z_shape', 'x_length/x_shape', 'y_length/y_shape', 'iou_on_xoy', 'is_rib',
                                     'distance_nearest_centroid', 'point_count', 'mean_z_distance_on_xoy', 'max_nonzero_internal']]
all_bone_info_df.to_csv('/home/jiangyy/projects/medical-rib/bone_df_csv/all_bone_info_df.csv')

