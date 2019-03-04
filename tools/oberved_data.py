
import pandas as pd
from skimage.measure import label
import skimage
import skimage.morphology as sm
from preprocessing.separated.ribs_obtain.util import arr_to_sparse_df
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology,feature

RIB_DF_CACHE_PATH="~/Desktop/logs/135402000150175/label_88785_collect_NOT_RIB.csv"
bone_df = pd.read_csv(RIB_DF_CACHE_PATH)
# print(bone_df.head(10))
z_max, x_max, y_max = bone_df['z'].max() + 1, bone_df['x'].max()+1, bone_df['y'].max()+1

from preprocessing.separated.ribs_obtain import util
image = util.sparse_df_to_arr([z_max, x_max, y_max], sparse_df=bone_df, fill_bool=False)

distance = ndi.distance_transform_edt(image) #距离变换
local_maxi =feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                   labels=image)   #寻找峰值
markers = ndi.label(local_maxi)[0] #初始标记点
labels = skimage.morphology.watershed(-distance, markers, mask=image) #基于距离变换的分水岭算法


label_arr = skimage.measure.label(labels, connectivity=1)


sparse_df, cluster_df = arr_to_sparse_df(label_arr=label_arr, sort=True, sort_key='c.count',
                                         keep_by_top=True, top_nth=10,
                                         keep_by_threshold=True, threshold_min=4000)

print(cluster_df)
from preprocessing.separated.ribs_obtain.util import plot_yzd
#print(sparse_df[sparse_df['c']==16])
#plot_yzd(sparse_df[sparse_df['c']==2], shape_arr=[z_max, y_max])