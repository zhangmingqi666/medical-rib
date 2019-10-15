
### ribs detection

+ when bones separated from each other, which included ribs, scapula, clavicle, skull and so on. We use rib detection model to classify 
whether bone is rib or not.
+ GBDT was chosen as classification model, whose features were kinds of index for every bone, eg, bone length, bone thickness

### feature engineering

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

### train our detection model

+ `gbdt_judge_rib.py`: train gbdt models using training dataSet.
+ `update_err_labels_and_aggregate_bone_info.py`: update error predicted targets and merge bone info into training dataSet.
+ `feature.pkl`: features used for training including target
+ `gbdt.pkl`: use pre-trained models to predict rib or not

### predict rib or not

+ predict

