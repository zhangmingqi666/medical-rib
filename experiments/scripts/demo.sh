#!/usr/bin/env bash

dcm_path=$1
FORMAT='dcm'
SLICING=1
RIB_DF_CACHE_PATH=null
BONE_INFO_PATH=null
out_put_prefix=null
RIBS_MODEL_WEIGHTS=./experiments/cfgs/gbdt.pkl

python3  ./preprocessing/separated/main.py  --use_pkl_or_dcm  ${FORMAT}  \
                                            --dcm_path  ${dcm_path}  \
                                            --keep_slicing  ${SLICING}  \
                                            --rib_df_cache_path  ${RIB_DF_CACHE_PATH} \
                                            --bone_info_path  ${BONE_INFO_PATH}  \
                                            --output_prefix  ${out_put_prefix}  \
                                            --rib_recognition_model_path  ${RIBS_MODEL_WEIGHTS}


ribs_df_cache_folder=
Voc2007_JPEGImages_folder=
python3  ./preprocessing/prepare_data/voc2007/to_ribs_dataset_voc2007.py    --in_folder_path  ${ribs_df_cache_folder}
                                                                            --output_independent_rib_folder  ${Voc2007_JPEGImages_folder}
                                                                            --output_format  '.jpg'



files=$(ls ${Voc2007_JPEGImages_folder})
for f in ${files}
do
    ./darknet detector test ./cfg/hurt_voc.data ./cfg/yolov3-voc.cfg ./backup/yolov3-voc.backup ${Voc2007_JPEGImages_folder}/${f}
done

