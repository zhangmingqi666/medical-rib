#!/usr/bin/env bash

# dataSet struct
# dataset
# label
# rib_df_cache
# pkl_cache
#
#
#
PY=python3
#dataSet_folder="/Users/jiangyy/projects/temp/medical-rib/dataSet/first20labeled"
dataSet_folder=$1

dataset_offset_df_file="offset_df.csv"
rib_type_file="rib_type_location.xls"
data_join_label_file="data_join_label.csv"
label_loc_type_info_file="label_loc_type_info.csv"
label_info_dir="label_info.csv"

label_dir="label"
rib_only_cache_dir="rib_df_cache"
voc_2007_dir="voc2007"
Annotations_dir="Annotations"
ImageSets_dir="ImageSets"
JPEGSImages_dir="JPEGSImages"

# input folders and files
label_folder=${dataSet_folder}/${label_dir}
rib_type_path=${dataSet_folder}/${rib_type_file}

# output folders and files
label_info_path=${dataSet_folder}/${label_info_dir}
data_join_label_path=${dataSet_folder}/${data_join_label_file}
dataset_offset_df_path=${dataSet_folder}/${dataset_offset_df_file}
label_loc_type_info_path=${dataSet_folder}/${label_loc_type_info_file}

# voc 2007 dirs
rib_only_cache_folder=${dataSet_folder}/${rib_only_cache_dir}
voc2007_folder=${dataSet_folder}/${voc_2007_dir}
voc2007_Annotations_folder=${voc2007_folder}/${Annotations_dir}
voc2007_ImageSets_folder=${voc2007_folder}/${ImageSets_dir}
voc2007_JPEGSImages_folder=${voc2007_folder}/${JPEGSImages_dir}

function re_mkdir_folder(){
    rm -rf $1
    mkdir -p $1
    echo "re mkdir $1"
}

function remove_file(){
    rm -rf $1
    echo "remove last results or intermediate files"
}

# re_mkdir_folder  ${rib_only_cache_folder}
re_mkdir_folder  ${voc2007_Annotations_folder}
re_mkdir_folder  ${voc2007_ImageSets_folder}
re_mkdir_folder  ${voc2007_JPEGSImages_folder}

remove_file  ${label_info_path}
remove_file  ${data_join_label_path}
remove_file  ${dataset_offset_df_path}
remove_file  ${label_loc_type_info_path}

echo "######## read nii ########"
# read nii and join with rib_type_location.xls
${PY}  src/read_label_info.py  --excel_path  ${rib_type_path}  \
                               --nii_folder  ${label_folder}  \
                               --output_path  ${label_info_path}

exit
echo "######## only one rib data join with label, id-R4 join id-338 ########"

${PY}  src/data_join_label.py  --rib_only_cache_folder  ${rib_only_cache_folder}  \
                               --label_info_path  ${label_info_path}  \
                               --data_join_label_path  ${data_join_label_path}

echo "######## split ribs_dataSet into independent picture and save offset shift ########"
${PY}  src/to_ribs_dataset_voc2007.py  --in_folder_path  ${rib_only_cache_folder}  \
                                       --output_independent_rib_folder  ${voc2007_JPEGSImages_folder}  \
                                       --dataset_offset_df_path  ${dataset_offset_df_path}  \
                                       --output_format  ".jpg"

echo "######## bounding box shift according to rib min ########"
${PY}  src/bounding_box_shift.py  --label_info_path  ${label_info_path}  \
                                  --dataset_offset_df_path  ${dataset_offset_df_path}  \
                                  --data_join_label_path  ${data_join_label_path}  \
                                  --label_loc_type_info_path  ${label_loc_type_info_path}

echo "######## generate voc2007 format: annotations ########"
${PY}  src/generate_xml_voc2007.py  --label_loc_type_info_path  ${label_loc_type_info_path}  \
                                    --voc2007_Annotations_folder  ${voc2007_Annotations_folder}

echo "######## generate voc2007 format: imageSets ########"
${PY}  src/write_imagesets_voc2007.py  --voc2007_Annotations_folder  ${voc2007_Annotations_folder}  \
                                       --voc2007_ImageSets_folder  ${voc2007_ImageSets_folder}

