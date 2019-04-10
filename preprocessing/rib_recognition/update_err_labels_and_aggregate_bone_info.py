
import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Update csv file')

parser.add_argument('--feature_csv_path', required=True, dest='feature_csv_path', action='store',
                    help='feature_csv_path')
parser.add_argument('--update_err_path', required=True, dest='update_err_path', action='store',
                    help='update_err_path')
parser.add_argument('--bone_feature_info_path', required=True, dest='bone_feature_info_path', action='store',
                    help='bone_feature_info_path')
args = parser.parse_args()

update_df = pd.read_csv(args.update_err_path, dtype={'ID': np.str, 'class_id': np.float})

df = pd.DataFrame({})
for f in os.listdir(args.feature_csv_path):
    ct_id = f.replace('.csv', '')
    f = "{}/{}.csv".format(args.feature_csv_path, ct_id)
    temp_df = pd.read_csv(f)

    value = update_df[update_df['ID'] == ct_id]['class_id'].values
    temp_df.loc[temp_df['class_id'].isin(value), 'target'] = 2.0
    df = df.append(temp_df)

df.to_csv(args.bone_feature_info_path, index=False)
