#!/usr/bin/env bash

pkl_dir=/home/jiangyy/projects/rib_20181214/medical-rib/dataset
output_dir=/home/jiangyy/projects/rib_20181214/medical-rib/temp_output
#python3 main.py --pkl_path "/home/jiangyy/projects/rib_20181212/medical-rib/cache/135402000550272.pkl" --output_prefix $output_dir/"hello_123"

#exit
files=$(ls $pkl_dir)
# array=("135402000565843.pkl" "135402000572309.pkl" "A1076955.pkl" "A1076956.pkl")
# array=("135402000409579.pkl" "A1077607.pkl")
array=("A1077090.pkl" "A1077414.pkl" "A1077120.pkl" "135402000150175.pkl")
for f in $array;
#for f in $files;
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
        echo "$folder"
        mkdir "$folder"

        if [[ -d "/home/jiangyy/Desktop/rib_df_input/$folder" ]]; then
            rm -r "/home/jiangyy/Desktop/rib_df_input/$folder"
        fi

        #mkdir "/home/jiangyy/Desktop/rib_df_input/$prefix"

        python3 main.py --pkl_path "$file_path" --output_prefix "$folder"/ > $output_dir/$prefix".log"
    fi
done
