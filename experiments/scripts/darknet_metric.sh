#!/usr/bin/env bash


python3 ./models/metric/reval_voc_py3.py ./models/darknet/results \
                                            --voc_dir ./data/voc2007 \
                                            --year 2007 \
                                            --image_set test \
                                            --classes ./models/darknet/data/hurt_voc.names
