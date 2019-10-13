
#!/usr/bin/env bash

# path=$(pwd)

feature_csv_dir=./data/bone_info_merges
update_err_path=./data/csv_files/update_err_bone_info.csv
bone_feature_info_path=./data/csv_files/bone_features_info.csv

gbdt_model_save_dir=./experiments/cfgs/gbdt.pkl
feature_list_save_dir=./experiments/cfgs/feature.pkl

echo "Start update and melt bone_info"
python3  -W ignore  ./preprocessing/rib_recognition/update_err_labels_and_aggregate_bone_info.py \
                                                            --feature_csv_path "$feature_csv_dir" \
                                                            --update_err_path "$update_err_path" \
                                                            --bone_feature_info_path "$bone_feature_info_path"
echo "Save dataset: $bone_feature_info_path"

echo "Start train GBDT model"
python3  -W ignore ./preprocessing/rib_recognition/gbdt_judge_rib.py --dataset_path "$bone_feature_info_path" \
                                                          --saved_gbdt_path "$gbdt_model_save_dir" \
                                                          --saved_feature_path "$feature_list_save_dir" # > train.log

echo "Saved GBDT model: $gbdt_model_save_dir"
echo "Saved feature list: $feature_list_save_dir"

