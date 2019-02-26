#!/usr/bin/env bash

DATA=$1
nii_folder=./data/nii_files_merges/
nii_loc_df_path=./data/csv_files/nii_loc_df.csv

case ${DATA} in
  updated48labeled_1.31)
    batch_data_names=("updated48labeled_1.31")
    ;;
  all_labeled)
    batch_data_names=("updated48labeled_1.31"     \
                      "added51-100labeled_2.21"   \
                      "added101-150labeled_2.21"  \
                      "added151-200labeled_2.21"  \
                      "added201-237labeled_2.21")
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

array=( "${batch_data_names[@]/#/${nii_folder}}" )

function join { local IFS="$1"; shift; echo "$*"; }
result=$(join " " ${array[@]})

echo "all data batches waiting to processing are ${result}"

# [1,1,1] or [z, z, z]
KEEP_SLICING=$2
python3 ./preprocessing/separated/nii_read/read_label_info.py   --nii_folder_list ${result} \
                                                                --output_path ${nii_loc_df_path} \
                                                                --keep_slicing ${KEEP_SLICING}