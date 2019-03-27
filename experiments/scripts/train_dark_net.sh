#!/usr/bin/env bash

echo "before train darknet, please"

./models/darknet/darknet detector train ./models/darknet_cfg/hurt_voc.data ./models/darknet_cfg/yolov3-voc.cfg ./models/darknet/darknet53.conv.74



