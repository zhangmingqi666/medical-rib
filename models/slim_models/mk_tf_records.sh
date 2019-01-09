#!/usr/bin/env bash

# location to where to save the TFRecord data.
TRAIN_DIR="/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/mk_tf_records/trainSet"
VALIDATION_DIR="/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/mk_tf_records/testSet"
OUTPUT_DIRECTORY="/Users/jiangyy/projects/temp/medical-rib/models/temp_data"
LABELS_FILE="/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/label.txt"
# build the preprocessing script.
cd /Users/jiangyy/workspace/models/research/inception
bazel build //inception:build_image_data --python_path=python3

# convert the data.
bazel-bin/inception/build_image_data \
  --train_directory="${TRAIN_DIR}" \
  --validation_directory="${VALIDATION_DIR}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --train_shards=32 \
  --validation_shards=8 \
  --num_threads=8