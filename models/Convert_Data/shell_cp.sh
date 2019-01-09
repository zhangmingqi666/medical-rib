#!/usr/bin/env bash

from_folder=/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/negtive_images
to_folder=/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/testSet/unfragment
move_file=/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/testSet/unfragment.txt
cat $move_file | while read line
do
    file_name=${from_folder}/${line}".jpg"
    echo $file_name
    cp $file_name $to_folder/
done