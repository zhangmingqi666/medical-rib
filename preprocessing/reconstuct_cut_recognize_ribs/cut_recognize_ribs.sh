#!/usr/bin/env bash

dataset_folder=${1}
pkl_dir=${dataset_folder}/"pkl_cache"
rib_df_dir=${dataset_folder}/"rib_df_cache"
rib_recognition_model_path="../Rib_Recognition_Model"
output_dir="../../Verify_logs/temp_output"

if [[ ! -d ${pkl_dir} ]]; then
     echo "${pkl_dir} not exist"
     exit 1
fi

if [[ ! -d ${rib_df_dir} ]]; then
     rm -rf ${rib_df_dir}
     mkdir -p ${rib_df_dir}
fi

files=$(ls ${pkl_dir})


for f in ${files}
do
    file_path=${pkl_dir}/${f}
    if [[ "$file_path" == *.pkl ]]
    then
        prefix=${f%%".pkl"}
        echo "cut $prefix data"
        folder=${output_dir}/${prefix}
        if [[ -d "$folder" ]]; then
            rm -rf "$folder"
        fi
        echo "Load rib recognition model from ${rib_recognition_model_path}"
        echo "save Verify_logs to ${folder}"
        echo "save rib_df_cache to ${rib_df_dir}"
        mkdir -p ${folder}

        python3 main.py --pkl_path ${file_path}  \
                        --output_prefix ${folder} \
                        --rib_recognition_model_path ${rib_recognition_model_path} \
                        --rib_df_cache_path ${rib_df_dir}  \
                        > $output_dir/$prefix".log" \

    fi
done
