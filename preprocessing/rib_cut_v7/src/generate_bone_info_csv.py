"""
@Author: Leaves
@Time: 17/01/2019
@File: generate_bone_info_csv.py
@Function: generate datafrom from separate csv feature files
"""
import os
import pandas as pd

file_path = '/home/jiangyy/projects/medical-rib/bone_df_csv'
files = os.listdir(file_path)

first_csv = pd.read_csv(open(os.path.join(file_path, files[0]), encoding='utf-8'))
features_list = list(first_csv.columns[1:])

all_bone_info_df = pd.DataFrame({}, columns=features_list)
for f in files:
    path = os.path.join(file_path, f)
    single_patient_bone = open(path, encoding='utf-8')
    temp_df = pd.read_csv(single_patient_bone)
    all_bone_info_df = all_bone_info_df.append(temp_df)

all_bone_info_df.reset_index(inplace=True)
all_bone_info_df = all_bone_info_df[features_list]
all_bone_info_df.to_csv('/home/jiangyy/projects/medical-rib/bone_df_csv/all_bone_info_df.csv')

