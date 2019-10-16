#!/usr/bin/env bash

ct_data_dir=/home/jiangyy/projects/medical-rib/data/dicom_files_merges/RIB_Batch1
demo_dir=/home/jiangyy/projects/medical-rib/data/demo_dir
files=$(ls ${ct_data_dir})

for f in ${files}
do
    echo "test ${f}"
    input_f=${ct_data_dir}/${f}
    ./experiments/scripts/demo.sh ${input_f} ${demo_dir}
done