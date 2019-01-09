#!/usr/bin/env bash

pkl_dir=/Users/jiangyy/projects/medical-rib/cache
output_dir=/Users/jiangyy/projects/medical-rib/temp_output
python3 main.py --pkl_path "/Users/jiangyy/projects/medical-rib/cache/135402000550272.pkl" --output_prefix $output_dir/"hello_123"

exit
files=$(ls $pkl_dir)
for f in $files;
do
    file_path=$pkl_dir/$f
    if [[ "$file_path" == *.pkl ]]
    then
        prefix=${f%%".pkl"}
        echo "cut $prefix data"
        folder=$output_dir/$prefix
        if [[ -d "$folder" ]]; then
            rm -r "$folder"
        fi
        mkdir "$folder"
        python3 main.py --pkl_path "$file_path" --output_prefix "$folder"/ > $output_dir/$prefix".log"
    fi
done