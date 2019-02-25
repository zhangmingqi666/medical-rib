#!/usr/bin/env bash

nii_folder="/Users/jiangyy/DataSet/rib_dataSet/nii_files_merges/"
batch_data_names=("updated48_dataset") # "added51-100_dataset" "added101-150_dataset" "added151-200_dataset" "added201-237_dataset")

nii_loc_df_path="/Users/jiangyy/DataSet/rib_dataSet/nii_loc_df.csv"

array=( "${batch_data_names[@]/#/${nii_folder}}" )
#echo ${array[@]}

function join { local IFS="$1"; shift; echo "$*"; }
result=$(join " " ${array[@]})

echo "all data batches waiting to processing are ${result}"

python3 src_read_nii_location/read_label_info.py  --nii_folder_list ${result} \
                                                  --output_path ${nii_loc_df_path} \
                                                  --keep_slicing