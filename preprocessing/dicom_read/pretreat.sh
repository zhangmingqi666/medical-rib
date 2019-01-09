#!/usr/bin/env bash

dcm_folder="/home/jiangyy/Desktop/CT_DATA"
dcm_df_path="temp_dataset_new_dcm.csv"

pkl_folder="/home/jiangyy/projects/medical-rib/dataset_100"


# read dcm path and its id
python3 main_read_dcm_path.py  --dcm_folder  ${dcm_folder}  --dcm_file_csv_path  ${dcm_df_path}

# transfer all dcm to array pkl for every patient
cat ${dcm_df_path} | tail -n +2 | while IFS=, read id dcm_path
do
    pkl_path=${pkl_folder}/${id}.pkl
    echo "$pkl_path"
    python3 main_create_pkl.py  --dcm_path  ${dcm_path}  --output_pkl_path  "${pkl_path}"
done

