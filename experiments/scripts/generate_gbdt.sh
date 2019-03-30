
#!/usr/bin/env bash

# path=$(pwd)

feature_csv_dir=./data/bone_info_merges
output_dataset_dir=./data/csv_files/all_bone_info_df.csv

gbdt_model_save_dir=./experiments/cfgs/gbdt.pkl
feature_list_save_dir=./experiments/cfgs/feature.pkl

echo "Start generate dataset"
python3  -W ignore  ./preprocessing/rib_recognition/generate_bone_info_csv.py --feature_csv_path "$feature_csv_dir" \
                                                                              --output_dataset_path "$output_dataset_dir"
echo "Save dataset: $output_dataset_dir"

echo "Start train GBDT model"
python3  -W ignore ./preprocessing/rib_recognition/gbdt_judge_rib.py --dataset_path "$output_dataset_dir" \
                                                          --saved_gbdt_path "$gbdt_model_save_dir" \
                                                          --saved_feature_path "$feature_list_save_dir" # > train.log

echo "Saved GBDT model: $gbdt_model_save_dir"
echo "Saved feature list: $feature_list_save_dir"

