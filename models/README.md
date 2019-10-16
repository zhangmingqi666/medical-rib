

### yolo-v3 operations

```
cd ${projects}/models

git clone https://github.com/pjreddie/darknet
cd darknet
vim Makefile

# update these three values.
GPU=1 # 0 if use cpu
CUDNN=1 # 0 if use cpu or cudnn not available
OPENCV=1 # 1 if use opencv else 0
NVCC=/path/to/nvcc

# build
make

cp darknet_cfg/cfg/* darknet/cfg/
cp darknet_cfg/data/hurt_voc.names darknet/data/

# train yolo-v3
./darknet detector train ./cfg/hurt_voc.data ./cfg/yolov3-voc.cfg ./darknet53.conv.74 -gpus 0,1,2,3
# test view
./darknet detector test ./cfg/hurt_voc.data ./cfg/yolov3-voc.cfg ./backup/yolov3-voc.backup ../../data/voc2007/JPEGImages/135402000404127-73082.jpg
# valid test saved in the results/comp4_det*.txt
python3 ./models/metric/reval_voc_py3.py ./models/darknet/results \
                                         --voc_dir ./data/voc2007 \
                                         --year 2007 \
                                         --image_set test \
                                         --classes ./models/darknet/data/hurt_voc.names
```


More information, we can refer to [yolo-v3 model](https://pjreddie.com/darknet/yolo/), [darknet configurations](https://zhuanlan.zhihu.com/p/35490655), [test commnands](https://blog.csdn.net/mieleizhi0522/article/details/79989754)