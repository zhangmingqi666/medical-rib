#!/usr/bin/env bash

file_path=/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/trainval.txt
iamge_from_folder=/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/images
image_to_folder=/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/before_augmented
cat ${file_path} | while IFS=, read id
do
   pic_path=${iamge_from_folder}/${id}.jpg
   mv ${pic_path} ${image_to_folder}/
done
