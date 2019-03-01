#!/usr/bin/env bash

DATA=$1
nii_folder=./data/nii_files_merges/


case ${DATA} in
  updated48labeled_1.31)
    batch_data_names=("updated48labeled_1.31")
    ;;
  added51-100labeled_2.21)
    batch_data_names=("added51-100labeled_2.21")
    nii_loc_df_path=./data/csv_files/nii_loc_df_added51-100.csv
    ;;
  added101-150labeled_2.21)
    batch_data_names=("added101-150labeled_2.21")
    nii_loc_df_path=./data/csv_files/nii_loc_df_added101-150.csv
    ;;
  added151-200labeled_2.21)
    batch_data_names=("added151-200labeled_2.21")
    nii_loc_df_path=./data/csv_files/nii_loc_df_added151-200.csv
    ;;
  added201-237labeled_2.21)
    batch_data_names=("added201-237labeled_2.21")
    nii_loc_df_path=./data/csv_files/nii_loc_df_added201-200.csv
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

