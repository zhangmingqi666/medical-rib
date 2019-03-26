#!/usr/bin/env bash

path=$1
files=$(ls ${path})
for file in ${files}:
do
    f_name=${path}/${file}
    if [[ -d ${f_name} ]]; then
      cnt=$(ls ${f_name} | grep $2 | wc -l)
      echo ${file}" "${cnt}
    fi
done