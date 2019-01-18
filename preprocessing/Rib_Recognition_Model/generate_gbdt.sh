#!/usr/bin/env bash

path=$(pwd)

feature_csv_dir=/home/jiangyy/projects/medical-rib/rib_feature_csv
output_dataset_dir=${path}/all_bone_info_df.csv

gbdt_model_save_dir=${path}/gbdt.pkl
feature_list_save_dir=${path}/feature.pkl

echo "Start generate dataset"
python3 -W ignore generate_bone_info_csv.py --feature_csv_path "$feature_csv_dir" --output_dataset_path "$output_dataset_dir"
echo "Save dataset: $output_dataset_dir"

echo "Start train GBDT model"
python3 -W ignore gbdt_judge_rib.py --dataset_path "$output_dataset_dir" --saved_gbdt_path "$gbdt_model_save_dir" \
                          --saved_feature_path "$feature_list_save_dir" > $path/"train.log"

echo "Saved GBDT model: $gbdt_model_save_dir"
echo "Saved feature list: $feature_list_save_dir"
