
![REFED_logo](.github/logo_refed_side_white.jpg)

## Refed

**Refed (Rib Extaction and Fracture Detection Model)** by [Youyou Jiang](jiangyy5318@gmail.com) and [Shiye Lei](leishiye@gmail.com). Our model has a great performance on **extracting ribs** and **detecting fracture** from CT images. The model is based on python 3.6. 

Refed is consist by two modules: **rib extraction module** and **fracture detection module**. *First*, we design algorithm based on computer vision for extracting ribs from CT images. *Second*, we design DNN based on [faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) for detecting fracture location with these ribs.


### Workflow


![workflow](.github/tech_route.jpeg)


add some details for them.


## Installation and Dependencies

+ Install tensorflow. It is required that you have access to GPUs, The code is tested with Ubuntu 16.04
Python 3.6+, CUDA 9.0.176 and cudnn 7.4.2.
+ Python dependencies (with `pip3 install`):
```
    tensorflow-gpu==1.12.0
    pandas==0.23.4
    scikit-learn==0.20.0
    numpy==1.16.2
    matplotlib==3.0.0
    PIL==Pillow
    lxml==4.2.5 
```
+ Clone the repository
```shell
    git clone https://github.com/jiangyy5318/medical-rib.git
```
+ Config darknet models
```shell
    cd ${Projects}/models
    git clone https://github.com/pjreddie/darknet
    cp models/darknet_cfg/yolov3-voc.cfg models/darknet/cfg/
    cp models/darknet_cfg/hurt_voc.data models/darknet/cfg/
    cp models/darknet_cfg/hurt_voc.names models/darknet/data/
```


## Demo and Test with pre-trained models

You can download pre-trained models, including GBDT model [HERE](https://drive.google.com/open?id=1_-dP4Y6wYDC5lqQ4uaIcXrAM-AHT_xd7), 
GBDT features [HERE](https://drive.google.com/open?id=1R8OkfLWniBhjFkAAYDlTWYwavt4dYaiB) and yolo-v3 models [HERE](added). Put `feature.pkl`, `gbdt.pkl` under the project root path (`${project}/experiments/cfgs`) and 
Put `?.pkl` under the project root path (`${projects}/experiments/cfgs`) 

```shell
    ./experiments/scripts/demo.sh [DCM_PATH]
    # DCM_PATH is folder path where CT slices existed.
```

## Train your own model

### Data Preparation

data save format, you need to refer to the data [待添加]

```shell
    ./experiments/scripts/nii_read.sh [DATA] [SLICING]
    ./experiments/scripts/dcm_read.sh [DATA]
    ./experiments/scripts/ribs_obtain.sh [LOGS_DIR] [FORMAT] [SLICING]
    ./experiments/scripts/prepare_data.sh
    # [DATA] in {updated48labeled_1.31, all_labeled} originated from different batches of data, has been defined in dcm_read.sh and nii_read.sh
    # [SLICING] in {0,1} is defined for interpolation interval, if 0, interval=1mm, else interval=slicing thickness 
    # [LOGS_DIR] used for debug
    # [FORMAT] '.dcm'
```

### train gbdt model (generated `feature.pkl` and `gbdt.pkl`)

step `./experiments/scripts/ribs_obtain.sh` will generate many separated bones and GBDT model will recognize all the ribs,
all the features for every bone will be saved in the path `./data/bone_info_merges`, you can added the misclassified bone in the
path `./data/csv_files/update_err_bone_info.csv` and run the below script.

```shell
    ./experiments/scripts/generate_gbdt.sh
```

### train your own yolo-v3 models

```shell
    wget ./darknet53.conv.74
    ./darknet detector train ./cfg/hurt_voc.data ./cfg/yolov3-voc.cfg ./darknet53.conv.74 -gpus 0,1,2,3
```
