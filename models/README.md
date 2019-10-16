

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

# train yolo-v3 from scrath
./darknet detector train ./cfg/hurt_voc.data ./cfg/yolov3-voc.cfg ./darknet53.conv.74 -gpus 0,1,2,3

# train yolo-v3 from pretrained model.

```


More information, we can refer to [yolo-v3 model](https://pjreddie.com/darknet/yolo/), [darknet configurations](https://zhuanlan.zhihu.com/p/35490655), [test commnands](https://blog.csdn.net/mieleizhi0522/article/details/79989754)