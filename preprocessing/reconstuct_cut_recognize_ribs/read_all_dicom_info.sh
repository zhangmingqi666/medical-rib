#!/usr/bin/env bash

# dataset folder
folder="/Users/jiangyy/DataSet/rib_dataSet/dicom_files_merges"
files=("updated48_dataset") # "added51-100_dataset" "added101-150_dataset" "added151-200_dataset" "added201-237_dataset")

# id, path csv
dcm_df_out_path="/Users/jiangyy/DataSet/rib_dataSet/dicom_info.csv"

# pkl output directory
pkl_folder="/Users/jiangyy/DataSet/rib_dataSet/updated48labeled_1.31/pkl_cache_bak"
spacing_df_path="/Users/jiangyy/DataSet/rib_dataSet/spacing_info.csv"


dcm_folders=${folder}/${files[0]} #" "${folder}/${files[1]}" "${folder}/${files[2]}" "${folder}/${files[3]}" "${folder}/${files[4]}

# read dcm path and its id
python3 src_reconstruct/main_read_dcm_path.py  --dcm_folder  ${dcm_folders}  --dcm_file_csv_path  ${dcm_df_out_path}

if [[ ${pkl_folder} == "" ]]; then
    echo "Not produce .pkl files"
    exit
fi

if [[ ! -d ${pkl_folder} ]]; then
     rm -rf ${pkl_folder}
     mkdir -p ${pkl_folder}
fi
python3 src_reconstruct/main_create_pkl.py  --dcm_df_path  ${dcm_df_out_path}  \
                                            --output_pkl_folder  ${pkl_folder}  \
                                            --spacing_df_path  ${spacing_df_path}
                                            #--keep_slicing
