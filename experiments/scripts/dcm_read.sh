#!/usr/bin/env bash

# some folders
DATA=$1
dcm_folder=./data/dicom_files_merges/

case ${DATA} in
  updated48labeled_1.31)
    batch_data_names=("updated48labeled_1.31")
    ;;
  all_labeled)
    batch_data_names=("updated48labeled_1.31" "added51-100labeled_2.21" "added101-150labeled_2.21" "added151-200labeled_2.21" "added201-237labeled_2.21")
    ;;
  *)
    echo "No dataset given"                        i
    exit
    ;;
esac

array=( "${batch_data_names[@]/#/${dcm_folder}}" )

function join { local IFS="$1"; shift; echo "$*"; }
result=$(join " " ${array[@]})


# scanning all the tree directory for every patient and find dcm nodes.
DCM_DF_OUT_PATH=./data/csv_files/dicom_info.csv
DCM_SCANNING=./preprocessing/separated/dcm_read/main_read_dcm_path.py
python3 ${DCM_SCANNING}  --dcm_folder  ${result}  \
                         --dcm_file_csv_path  ${DCM_DF_OUT_PATH}


# pkl output directory
PKL_FOLDER=$2
echo "pkl_folder is = "${PKL_FOLDER}
SPACING_DF_PATH=./data/csv_files/spacing_info.csv

# read dcm path and its id and reconstruct 3d image

if [[ ${PKL_FOLDER} == "" ]]; then
    echo "Not produce .pkl files"
    exit
fi

if [[ ! -d ${PKL_FOLDER} ]]; then
     rm -rf ${PKL_FOLDER}
     mkdir -p ${PKL_FOLDER}
fi

KEEP_SLICING=$3
echo "keep slicing is = "${KEEP_SLICING}
python3 ./preprocessing/separated/dcm_read/main_create_pkl.py  --dcm_df_path  ${DCM_DF_OUT_PATH}      \
                                                               --output_pkl_folder  ${PKL_FOLDER}     \
                                                               --spacing_df_path  ${SPACING_DF_PATH}  \
                                                               --keep_slicing  ${KEEP_SLICING}
