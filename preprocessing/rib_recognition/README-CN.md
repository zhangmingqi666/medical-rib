## 肋骨识别模型
- 一张CT图片经过肋骨切割后形成多根分离的骨头，其中包括肋骨、肩胛骨、锁骨、头骨等等，我们通过肋骨识别模型来判断骨头是否为肋骨.
- 肋骨识别模型的主体为GBDT，通过输入每根骨头的各项物理指标，例如骨头的长短、粗细等等来判断其是否为肋骨.

### 特征工程

+ `z_centroid/z_shape` : relative center position on axis z
+ `x_centroid/x_shape` : relative center position on axis x
+ `y_centroid/y_shape` : relative center position on axis y
+ `z_max/z_shape` : relative maximum position on axis z
+ `x_max/x_shape` : relative maximum position on axis x
+ `y_max/y_shape` : relative maximum position on axis y
+ `z_length/z_shape` : relative length on axis z
+ `x_length/x_shape` : relative length on axis x
+ `y_length/y_shape` : relative length on axis y
+ `mean_z_distance_on_xoy` : mean distance on axis z group by xoy
+ `std_z_distance_on_xoy` : std distance on axis z group by xoy
+ `std_z_distance_div_mean_z_distance` : div the two above
+ `mean_x_distance_on_zoy` : mean distance on axis x group by zoy
+ `std_x_distance_on_zoy` : std distance on axis x group by zoy
+ `std_x_distance_div_mean_x_distance` : div the two above
+ `mean_y_distance_on_xoz` : mean distance on axis y group by xoz
+ `std_y_distance_on_xoz` : std distance on axis y group by xoz
+ `std_y_distance_div_mean_y_distance` : div the two above
+ `iou_on_xoy` : rib projected on xoy / (x_length * y_length)
+ `distance_nearest_centroid` : The nearest distance between Centroid and points
+ `point_count` : point count per bone
+ `class_id` : used not for training but for updating error labels
+ `target` : label

### 训练模型

+ `gbdt_judge_rib.py`: 使用训练数据训练GBDT模型;
+ `update_err_labels_and_aggregate_bone_info.py`: 更改预测错误的类标并将其合成训练数据;
+ `feature.pkl`: 用于训练的特征名包括target
+ `gbdt.pkl`: 预训练的模型

### 预测

+ predict
