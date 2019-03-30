#!/usr/bin/env bash

path=$1
files=$(ls ${path})
#for file in ${files}:
#do
#    f_name=${path}/${file}
#    if [[ -d ${f_name} ]]; then
#      cnt=$(ls ${f_name} | grep $2 | wc -l)
#      echo ${file}" "${cnt}
#    fi
#done

for file in ${files}:
do
    f_name=${path}/${file}
    cnt=$(cat ${f_name} | tail -n +1 | awk -F',' '{print $1}' | sort | uniq -c | wc -l)
    echo ${file}" "${cnt}
done