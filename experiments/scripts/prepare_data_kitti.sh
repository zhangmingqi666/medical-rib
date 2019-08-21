#!/usr/bin/env bash

# 需要对after_join.csv 做出注释.
NII_LOC_PATH=./data/csv_files/nii_loc_df.csv
RIB_TYPE_LOCATION_PATH=./data/csv_files/rib_type_location.xls
RIB_DF_CACHE_DIR=./data/ribs_df_cache

# temp
OS=`uname -s`
#if [[ ${OS} == "Linux"  ]];then
#    RIB_DF_CACHE_DIR=./data/ribs_df_cache.last
#fi

JOIN_LABEL_PATH=./data/csv_files/join_label.csv

kitti_folder=./data/kitti
kitti_depth_folder=${kitti_folder}/depth
kitti_image_folder=${kitti_folder}/image
kitti_calib_folder=${kitti_folder}/calib
kitti_label_folder=${kitti_folder}/label_dimension

function re_mkdir_folder(){
    rm -rf $1
    mkdir -p $1
    echo "re mkdir $1"
}

re_mkdir_folder  ${kitti_depth_folder}
re_mkdir_folder  ${kitti_image_folder}
re_mkdir_folder  ${kitti_calib_folder}
re_mkdir_folder  ${kitti_label_folder}


# join all the rib, bounding box, excel_df
#python3 ./preprocessing/prepare_data/join_xls_nii_rib.py  --ribs_df_cache_folder  ${RIB_DF_CACHE_DIR} \
#                                                          --nii_loc_df_path  ${NII_LOC_PATH}  \
#                                                          --rib_type_location_path  ${RIB_TYPE_LOCATION_PATH}  \
#                                                          --data_join_label_path  ${JOIN_LABEL_PATH}
# do not need to eliminate offsets of label and point cloud

python3 ./preprocessing/prepare_data/kitti/write_kitti.py \
              --ribs_df_cache_folder  ${RIB_DF_CACHE_DIR} \
              --nii_loc_df_path  ${NII_LOC_PATH} \
              --rib_type_location_path  ${RIB_TYPE_LOCATION_PATH} \
              --data_join_label_path ${JOIN_LABEL_PATH}

#if [[ ${OS} == "Darwin"  ]];then
#    sed -i 's/Users/home/g' ${Voc2007_folder}/*.txt
#fi
