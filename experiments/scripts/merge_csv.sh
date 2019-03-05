#!/usr/bin/env bash


Prefix=$1
output_path=$2
rm -rf ${output_path}

files=$(ls ${Prefix})
echo ${files}
for f in ${files}
do
    echo ${f}
    if [[ ! -f "$output_path" ]]; then
        echo "hhhhh"
        touch ${output_path}
        cat ${f} > ${output_path}
    else
        echo "lllll"
        cat ${f} | tail -n +2 >> ${output_path}
    fi
done

