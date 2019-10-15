#!/usr/bin/env bash

Max_date=""
Min_date=""
files=$(ls ${1})
for f in ${files}
do
    f_path=${1}/${f}
    temp_date=$(date -r rename.sh +"%Y-%m-%d %H:%M:%S")
    if [[ ${Max_date} = "" ]]; then
        Max_date=${temp_date}
        Min_date=${temp_date}
    fi

    if [[ ${temp_date} -gt ${Max_date} ]]; then
        Max_date=${temp_date}
    fi

    if [[ ${temp_date} -lt ${Min_date} ]]; then
        Min_date=${temp_date}
    fi
done


echo "The Min time is ${Min_date}"
echo "The Max time is ${Max_date}"