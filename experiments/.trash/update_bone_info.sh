#!/usr/bin/env bash

UPDATE_FILE=$1
BONE_INFO_PATH=$2

cat ${UPDATE_FILE} | tail -n +2 | while IFS=, read id class_id
do
    echo $id $class_id
    # cat ${BONE_INFO_PATH}/${id}".csv"
    awk  -v a=$class_id 'BEGIN{FS=OFS=","} {if ($26==a) $25=2.0}1' ${BONE_INFO_PATH}/${id}".csv" > temp.csv && mv temp.csv ${BONE_INFO_PATH}/${id}".csv"
done