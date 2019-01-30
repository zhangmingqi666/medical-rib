#!/usr/bin/env bash

dataset_folder=$1
dcm_df_path="temp_dataset_new_dcm.csv"
dcm_folder=$dataset_folder/"dataset"
pkl_folder=$dataset_folder/"pkl_cache"

if [ ! -d ${dcm_folder} ]; then
     echo "$dcm_folder not exist"
     exit 1
fi

if [ ! -d ${pkl_folder} ]; then
     rm -rf ${pkl_folder}
     mkdir -p ${pkl_folder}
fi

# read dcm path and its id
python3 main_read_dcm_path.py  --dcm_folder  ${dcm_folder}  --dcm_file_csv_path  ${dcm_df_path}

# transfer all dcm to array pkl for every patient
cat ${dcm_df_path} | tail -n +2 | while IFS=, read id dcm_path
do
    pkl_path=${pkl_folder}/${id}.pkl
    echo "$pkl_path"
    python3 main_create_pkl.py  --dcm_path  ${dcm_path}  --output_pkl_path  "${pkl_path}"
done

rm -rf ${dcm_df_path}
