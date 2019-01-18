## 肋骨识别模型
- 一张CT图片经过肋骨切割后形成多根分离的骨头，其中包括肋骨、肩胛骨、锁骨、头骨等等，我们通过肋骨识别模型来判断骨头是否为肋骨.
- 肋骨识别模型的主体为GBDT，通过输入每根骨头的各项物理指标，例如骨头的长短、粗细等等来判断其是否为肋骨.
### 数据生成
- 所用的骨头的各项指标（Feature）：
  - `z_centroid/z_shape` : 骨头质心的Z坐标值 / CT图片的Z轴长度
  - `x_centroid/x_shape` :
  - `y_centroid/y_shape` :
  - `z_max/z_shape` : 骨头所有点的Z坐标最大值 / CT图片的Z轴长度
  - `x_max/x_shape` :
  - `y_max/y_shape` : 
  - `z_length/z_shape` : 骨头所有点的Z坐标最大值与最小值之差 / CT图片的Z轴长度
  - `x_length/x_shape` :
  - `y_length/y_shape` :
  - `mean_z_distance_on_xoy` : 以xoy为底，z方向上骨头的平均厚度
  - `std_z_distance_on_xoy` : z方向上骨头厚度的方差
  - `std_z_distance_div_mean_z_distance` : z方向上骨头厚度方差 / z方向上骨头厚度平均值
  - `median_z_distance_on_xoy` : z方向上骨头厚度的中位数
  - `skew_z_distance_on_xoy` : z方向上骨头厚度的斜度（skew）
  - `kurt_z_distance_on_xoy` : z方向上骨头厚度的峰度（kurt）
  - `quantile_down_z_distance_on_xoy` : z方向上骨头厚度的下分位数
  - `quantile_on_z_distance_on_xoy` : z方向上骨头厚度的上分位数
  - `mean_x_distance_on_zoy` : 
  - `std_x_distance_on_zoy` : 
  - `std_x_distance_div_mean_x_distance` : 
  - `median_x_distance_on_zoy` : 
  - `skew_x_distance_on_zoy` : 
  - `kurt_x_distance_on_zoy` : 
  - `quantile_down_x_distance_on_zoy` : 
  - `quantile_on_x_distance_on_zoy` : 
  - `mean_y_distance_on_zox` : 
  - `std_y_distance_on_zox` : 
  - `std_y_distance_div_mean_y_distance` : 
  - `median_y_distance_on_zox` : 
  - `skew_y_distance_on_zox` : 
  - `kurt_y_distance_on_zox` : 
  - `quantile_down_y_distance_on_zox` : 
  - `quantile_on_y_distance_on_zox` : 
  - `iou_on_xoy` : 将骨头投影到xoy平面度的IOU
  - `distance_nearest_centroid` : 离骨头质心最近的点的距离
  - `point_count` : 骨头中点的数量
  - `max_nonzero_internal` : 肋骨之间的gap
  - `class_id` : 做skimage.label时骨头的类名，仅仅用作数据集label标定，不用做训练和预测
  - `target` : （label）1代表骨头是肋骨，2代表不是肋骨
- 原始数据生成：
  - 我们在[preprocessing.rib_cut_v7.src.Bone_decompose.py](https://github.com/jiangyy5318/medical-rib/blob/master/preprocessing/rib_cut_v7/src/Bone_decompose.py)中的collect_rib()中将切割后的CT图片中的每块骨头的features都存储到[rib_feature_csv](https://github.com/jiangyy5318/medical-rib/tree/master/preprocessing/Rib_Recognition_Model/rib_feature_csv)中.
  - 我们已经在[rib_feature_csv](https://github.com/jiangyy5318/medical-rib/tree/master/preprocessing/Rib_Recognition_Model/rib_feature_csv)中给出了生成骨头feature数据的一些示例.
### 制作用于训练肋骨识别模型的数据集
- [generate_bone_from_csv.py](https://github.com/jiangyy5318/medical-rib/blob/master/preprocessing/Rib_Recognition_Model/generate_bone_info_csv.py)将上一步生成的各个CT图像所包含的骨头features连接起来，生成一个总的[all_bone_info_df.csv](https://github.com/jiangyy5318/medical-rib/blob/master/preprocessing/Rib_Recognition_Model/all_bone_info_df.csv)以投入肋骨识别模型进行训练.
- 我们已经在[all_bone_info_df.csv](https://github.com/jiangyy5318/medical-rib/blob/master/preprocessing/Rib_Recognition_Model/all_bone_info_df.csv)中给出了一个用于训练肋骨识别模型的数据集的示例.
### 训练以及保存肋骨识别模型
- 模型训练位于[gbdt_judge.py](https://github.com/jiangyy5318/medical-rib/blob/master/preprocessing/Rib_Recognition_Model/gbdt_judge_rib.py)中.
- 模型主体为GBDT.
- [gbdt_judge.py](https://github.com/jiangyy5318/medical-rib/blob/master/preprocessing/Rib_Recognition_Model/gbdt_judge_rib.py) 中读入上一步生成的[all_bone_info_df.csv](https://github.com/jiangyy5318/medical-rib/blob/master/preprocessing/Rib_Recognition_Model/all_bone_info_df.csv)数据集，将'target'作为label，将去除'target'属性的数据作为samples，投入sklearn中的gbdt model进行训练.
- 将训练完后的GBDT与FEATURE LIST以pkl的形式存储下来.
**Note**: FEATURE LIST中不含有`target`以及`class_id`
- 我们给出了已经训练完成的`肋骨识别模型`[gbdt.pkl](https://github.com/jiangyy5318/medical-rib/blob/master/preprocessing/Rib_Recognition_Model/gbdt.pkl)以及其对应的`FEATURE LIST` [feature.pkl](https://github.com/jiangyy5318/medical-rib/blob/master/preprocessing/Rib_Recognition_Model/feature.pkl).
### 肋骨识别模型的使用
- 读入肋骨识别模型：
```python
# load gbdt model and feature list
GBDT = joblib.load('/home/jiangyy/projects/temp/medical-rib/gbdt_model/gbdt.pkl')
FEATURE_LIST = joblib.load('/home/jiangyy/projects/temp/medical-rib/gbdt_model/feature.pkl')
```
- 使用肋骨识别模型进行预测：
```python
# initialize a bone
single_bone = Bone(bone_data=temp_sparse_df, arr_shape=bone_prior.get_prior_shape(), spine_width=100, prior_zoy_center_y_axis_line_df=bone_prior.get_zoy_symmetric_y_axis_line_df())
# get features of the bone
temp_single_bone_feature = single_bone.get_rib_feature_for_predict()
# recognize that if the bone is a rib
if GBDT.predict([[temp_single_bone_feature[i] for i in FEATURE_LIST]]):
    temp_single_bone_feature['target'] = 1
else:
    temp_single_bone_feature['target'] = 0
```
