#!/usr/bin/env bash

train_image_classifier_path=/Users/jiangyy/workspace/models/research/slim
DATASET_DIR=/Users/jiangyy/projects/temp/medical-rib/models/temp_data
TRAIN_DIR=tmp_logs
cd ${train_image_classifier_path}
python3 train_image_classifier.py --train_dir=${TRAIN_DIR} \
                                  --dataset_name=bone_net \
                                  --dataset_split_name=train \
                                  --dataset_dir=${DATASET_DIR} \
                                  --model_name=alexnet