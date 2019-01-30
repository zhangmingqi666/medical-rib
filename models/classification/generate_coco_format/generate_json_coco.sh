#!/usr/bin/env bash

label_loc_type_info_path=/Users/jiangyy/projects/medical-rib/dataSet/first20labeled/label_loc_type_info.csv
coco_json_file=coco.json
python3  generate_json_coco.py  --label_loc_type_info_path  ${label_loc_type_info_path}  \
                                --coco_json_file  ${coco_json_file}