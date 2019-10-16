#!/usr/bin/env bash

input_f=$1 # dcm folder or pkl path
patient_id='patient_id'
if [[ "$input_f" == *.pkl ]]
then
    FORMAT=pkl
    patient_id=${input_f%%".pkl"}
    echo "input format is .pkl"
elif [[ -d ${input_f} ]]
then
    files=$(ls ${input_f})
    dcm_exist=0
    for temp in ${files}
    do
        file_path=${input_f}/${temp}
        if [[ "$file_path" == *.dcm ]]
        then
            dcm_exist=1
        fi
    done

    if [[ ${dcm_exist} -eq 1 ]];
    then
        echo "input format is .dcm"
        FORMAT=dcm
        patient_id=${input_f##*/}
    else
        echo "no available dcm files in input folder"
        exit 1
    fi
else
    echo "input data format is error, you need input dcm folder or matrix pickle"
    exit 1
fi

#FORMAT=dcm
#dcm_path=$1  # input dcm path
demo_dir=$2
SLICING=1

RIBS_MODEL_WEIGHTS=./experiments/cfgs
RIB_DF_CACHE_DIR=${demo_dir}/ribs_df_cache
Voc2007_JPEGSImages_folder=${demo_dir}/voc_test_data

rm -rf ${RIB_DF_CACHE_DIR} && mkdir -p ${RIB_DF_CACHE_DIR}

rm -rf ${Voc2007_JPEGSImages_folder} && mkdir -p ${Voc2007_JPEGSImages_folder}

#RIB_DF_CACHE_PATH=${RIB_DF_CACHE_DIR}/"predict.csv"

python3  ./preprocessing/separated/main.py  --use_pkl_or_dcm  ${FORMAT}  \
                                            --dcm_path  ${input_f}  \
                                            --pkl_path  ${input_f}  \
                                            --keep_slicing  ${SLICING}  \
                                            --rib_df_cache_path  ${RIB_DF_CACHE_DIR}/${patient_id}".csv" \
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
    ./darknet detector test ./cfg/hurt_voc.data ./cfg/yolov3-voc.cfg ./backup/yolov3-voc_final.weights ${Voc2007_JPEGSImages_folder}/${f}
done

echo "predicted ok"
