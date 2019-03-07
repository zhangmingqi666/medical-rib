#!/usr/bin/env bash

path=/Users/jiangyy/Desktop/gbdt_model_data
FROM_RIB_DF_CACHE_DIR=${path}/ribs_df_cache
FROM_BONE_INFO_DIR=${path}/bone_info_merges
FROM_LOGS_DIR=${path}/logs

OUTPUT=${path}/output
mkdir -p ${OUTPUT}
maxsize=$((1024*10))
files=$(ls ${FROM_RIB_DF_CACHE_DIR})
for f in ${files}
do
    ct_id=${f%%".csv"}
    f_path=${FROM_RIB_DF_CACHE_DIR}/${f}

    file_size=`ls -l ${f_path} | awk '{ print $5 }'`

    if [[ ${file_size} -gt ${maxsize} ]]
    then
        output_ct=${OUTPUT}/${ct_id}
        mkdir ${output_ct}
        cp ${FROM_RIB_DF_CACHE_DIR}/${f} ${output_ct}/
        cp ${FROM_BONE_INFO_DIR}/${f} ${OUTPUT}/
        cp ${FROM_LOGS_DIR}/${ct_id}/"label*" ${output_ct}/
    else
        echo "${file_size} < ${maxsize}"
    fi

done