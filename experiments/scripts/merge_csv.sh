#!/usr/bin/env bash


Prefix=$1
output_path=$2
rm -rf ${output_path}


for f in ${Prefix}*
do
    if [[ ! -f "$output_path" ]]; then
        cat ${f} > ${output_path}
    else
        cat ${f} | tail -n +2 >> ${output_path}
    fi
done

