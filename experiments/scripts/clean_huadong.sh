#!/usr/bin/env bash


function clean_batch_nii() {
    # delete files:
    #       not start with [a-zA-Z1-9]
    #       not end with .nii
    # delete all the str "fudan-huadong-" in file or folder names
    for  file in  $(ls ${1});
    do
        f_path=${1}/${file}
        rename -f 's/fudan-huadong-//' ${f_path}/*

        for sub_file in $(ls ${f_path});
        do
            sub_f_path=${f_path}/${sub_file}
            if [[ ${sub_file} =~ ^[a-zA-Z1-9].*nii$ ]]
            then
                echo "${sub_file} match, reserved"
            else
                echo "${sub_file} not match, discarded"
                rm -rf ${sub_f_path}
            fi
        done
    done
}


nii_folder=./data/nii_files_merges

for file in $(ls ${nii_folder});
do
    echo "clean ${file}"
    clean_batch_nii ${nii_folder}/${file}
done




