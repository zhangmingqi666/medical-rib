#!/usr/bin/env bash

dataset_folder=$1
dcm_df_path="temp_dataset_new_dcm.csv"
dcm_folder=${dataset_folder}/"dataset"
pkl_folder=${dataset_folder}/"pkl_cache"
spacing_df_path=${dataset_folder}/"spacing_df_path.csv"

if [[ ! -d ${dcm_folder} ]]; then
     echo "$dcm_folder not exist"
     exit 1
fi

if [[ ! -d ${pkl_folder} ]]; then
     rm -rf ${pkl_folder}
     mkdir -p ${pkl_folder}
fi

# read dcm path and its id
python3 main_read_dcm_path.py  --dcm_folder  ${dcm_folder}  --dcm_file_csv_path  ${dcm_df_path}


python3 main_create_pkl.py  --dcm_df_path  ${dcm_df_path}  \
                            --output_pkl_folder  ${pkl_folder}  \
                            --spacing_df_path  ${spacing_df_path}

rm -rf ${dcm_df_path}
