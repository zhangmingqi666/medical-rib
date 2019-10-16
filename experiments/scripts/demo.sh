#!/usr/bin/env bash

PY=python3.6
input_f=$1 # dcm folder or pkl path
patient_id='patient_id'
if [[ "$input_f" == *.pkl ]]
then
    FORMAT=pkl
    file_name=${input_f##*/}
    patient_id=${file_name%%".pkl"}
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
Pkl_cache_folder=${demo_dir}/pkl_cache
Predict_folder=${demo_dir}/voc_test_predict

function mkdir_if_not_exist(){
    if [[ ! -d "$1" ]]; then
        mkdir -p $1
    fi
}

mkdir_if_not_exist  ${RIB_DF_CACHE_DIR}
mkdir_if_not_exist  ${Voc2007_JPEGSImages_folder}
mkdir_if_not_exist  ${Pkl_cache_folder}
mkdir_if_not_exist  ${Predict_folder}

#rm -rf ${Voc2007_JPEGSImages_folder} && mkdir -p ${Voc2007_JPEGSImages_folder}
#rm -rf ${Predict_folder} && mkdir -p ${Predict_folder}


${PY}  ./preprocessing/separated/main.py  --use_pkl_or_dcm  ${FORMAT}  \
                                            --dcm_path  ${input_f}  \
                                            --pkl_path  ${Pkl_cache_folder}/${patient_id}".pkl"  \
                                            --keep_slicing  ${SLICING}  \
                                            --rib_df_cache_path  ${RIB_DF_CACHE_DIR}/${patient_id}".csv" \
                                            --rib_recognition_model_path  ${RIBS_MODEL_WEIGHTS}
echo "separated ok"

Voc2007_JPEGSImages_folder_for_patient=${Voc2007_JPEGSImages_folder}/${patient_id}
Predict_folder_for_patient=${Predict_folder}/${patient_id}

rm -rf ${Voc2007_JPEGSImages_folder_for_patient} && mkdir -p ${Voc2007_JPEGSImages_folder_for_patient}
rm -rf ${Predict_folder_for_patient} && mkdir -p ${Predict_folder_for_patient}

${PY}  ./preprocessing/prepare_data/voc2007/to_ribs_dataset_voc2007.py    --in_folder_path  ${RIB_DF_CACHE_DIR}/${patient_id}".csv" \
                                                                          --output_independent_rib_folder  ${Voc2007_JPEGSImages_folder_for_patient} \
                                                                          --output_format  '.jpg'
echo "ribs saved to "${Voc2007_JPEGSImages_folder_for_patient}

cd models/darknet
files=$(ls ${Voc2007_JPEGSImages_folder_for_patient})
for f in ${files}
do
    ./darknet detector test ./cfg/hurt_voc.data ./cfg/yolov3-voc.cfg ./backup/yolov3-voc_final.weights ${Voc2007_JPEGSImages_folder_for_patient}/${f}
    mv predictions.jpg ${Predict_folder_for_patient}/${f}
done

echo "predicted ok"
