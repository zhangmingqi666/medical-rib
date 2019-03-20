#!/usr/bin/env bash

FORMAT=dcm
dcm_path=$1
SLICING=1
RIB_DF_CACHE_PATH=..
BONE_INFO_PATH=..
out_put_prefix=..
RIBS_MODEL_WEIGHTS=..

python3  ./preprocessing/separated/main.py  --use_pkl_or_dcm  ${FORMAT}  \
                                            --dcm_path  ${dcm_path}  \
                                            --keep_slicing  ${SLICING}  \
                                            --rib_df_cache_path  ${RIB_DF_CACHE_PATH} \
                                            --bone_info_path  ${BONE_INFO_PATH}  \
                                            --output_prefix  ${out_put_prefix}  \
                                            --rib_recognition_model_path  ${RIBS_MODEL_WEIGHTS}  \
                                            > ${out_put_prefix}".log"


# 这里文件和文件夹不对应.
python3  ./preprocessing/prepare_data/voc2007/to_ribs_dataset_voc2007.py  --in_folder_path  ${RIB_DF_CACHE_DIR}  \
                                         --output_independent_rib_folder  ${Voc2007_JPEGSImages_folder}  \
                                         --output_format  ".jpg"