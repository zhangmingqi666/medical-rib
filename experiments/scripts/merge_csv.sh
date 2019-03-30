#!/usr/bin/env bash


Prefix=$1
echo "input" ${Prefix}
output_path=$2
echo "output" ${output_path}
rm -rf ${output_path}

files=$(ls ${Prefix})

for f in ${files}
do
    f_name=${Prefix}/${f}
    if [[ ! -f ${output_path} ]]; then
        touch ${output_path}
        cat ${f_name} | head -n 1 > ${output_path}
    fi
    cat ${f_name} | tail -n +2 >> ${output_path}
done

