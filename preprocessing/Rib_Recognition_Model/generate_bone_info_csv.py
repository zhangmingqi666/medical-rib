"""
@Author: Leaves
@Time: 17/01/2019
@File: generate_bone_info_csv.py
@Function: generate datafrom from separate csv feature files
"""
import os
import pandas as pd


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Search some files')

    parser.add_argument('--feature_csv_path', required=True, dest='feature_csv_path', action='store', help='feature_csv_path')
    parser.add_argument('--output_dataset_path', required=True, dest='output_dataset_path', action='store', help='output_dataset_path')
    args = parser.parse_args()

    files = os.listdir(args.feature_csv_path)
    first_csv = pd.read_csv(open(os.path.join(args.feature_csv_path, files[0]), encoding='utf-8'))
    features_list = list(first_csv.columns[1:])

    all_bone_info_df = pd.DataFrame({}, columns=features_list)
    for f in files:
        path = os.path.join(args.feature_csv_path, f)
        single_patient_bone = open(path, encoding='utf-8')
        temp_df = pd.read_csv(single_patient_bone)
        all_bone_info_df = all_bone_info_df.append(temp_df)

    all_bone_info_df.reset_index(inplace=True)
    all_bone_info_df = all_bone_info_df[features_list]
    all_bone_info_df.to_csv(args.output_dataset_path)


if __name__ == '__main__':
    main()
