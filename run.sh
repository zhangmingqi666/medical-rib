#!/usr/bin/env bash


DATA=updated48labeled_1.31
KEEP_SLICING=1
LOGS_DIR=$1
FORMAT=dcm
# DATA=all_labeled
# read nii file


source activate venv

./experiments/scripts/nii_read.sh ${DATA} ${KEEP_SLICING}

# scanning dcm nodes
./experiments/scripts/dcm_read.sh ${DATA}

# ribs obtain
./experiments/scripts/ribs_obtain.sh ${LOGS_DIR} ${FORMAT} ${KEEP_SLICING}

# rib match


# make data

source deactivate