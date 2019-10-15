#!/usr/bin/env bash

FORMAT=dcm
dcm_path=$1  # input dcm path
demo_dir=$2
SLICING=1

RIBS_MODEL_WEIGHTS=./experiments/cfgs
RIB_DF_CACHE_DIR=${demo_dir}/ribs_df_cache
BONE_INFO_DIR=${demo_dir}/bone_info_merges
OUTPUT_DIR=${demo_dir}/logs_dir
Voc2007_JPEGSImages_folder=${demo_dir}/voc_test_data

rm -rf ${RIB_DF_CACHE_DIR} && mkdir -p ${RIB_DF_CACHE_DIR}
rm -rf ${BONE_INFO_DIR} && mkdir -p ${BONE_INFO_DIR}
rm -rf ${OUTPUT_DIR} && mkdir -p ${OUTPUT_DIR}
rm -rf ${Voc2007_JPEGSImages_folder} && mkdir -p ${Voc2007_JPEGSImages_folder}

out_put_prefix=${OUTPUT_DIR}/"predict"
BONE_INFO_PATH=${BONE_INFO_DIR}/"predict.csv"
RIB_DF_CACHE_PATH=${RIB_DF_CACHE_DIR}/"predict.csv"

python3  ./preprocessing/separated/main.py  --use_pkl_or_dcm  ${FORMAT}  \
                                            --dcm_path  ${dcm_path}  \
                                            --keep_slicing  ${SLICING}  \
                                            --rib_df_cache_path  ${RIB_DF_CACHE_PATH} \
                                            --bone_info_path  ${BONE_INFO_PATH}  \
                                            --output_prefix  ${out_put_prefix}  \
                                            --rib_recognition_model_path  ${RIBS_MODEL_WEIGHTS}
echo "separated ok"

python3  ./preprocessing/prepare_data/voc2007/to_ribs_dataset_voc2007.py    --in_folder_path  ${RIB_DF_CACHE_DIR} \
                                                                            --output_independent_rib_folder  ${Voc2007_JPEGSImages_folder} \
                                                                            --output_format  '.jpg'
echo "ribs saved to "${Voc2007_JPEGSImages_folder}

cd models/darknet
files=$(ls ${Voc2007_JPEGSImages_folder})
for f in ${files}
do
    ./darknet detector test ./cfg/hurt_voc.data ./cfg/yolov3-voc.cfg ./backup/yolov3-voc.backup ${Voc2007_JPEGSImages_folder}/${f}
done

echo "predicted ok"
