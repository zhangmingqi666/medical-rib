


# 二维:
# 原图大小.
# 线性拟合.

# def transfer_2d_image(multi_rib_df, left):

import pandas as pd
import numpy as np
from skimage.measure import label
import skimage
path = "/Users/jiangyy/projects/medical-rib/experiments/logs/135402000409413/label_119_collect_NOT_RIB.csv"
df = pd.read_csv(path)
map2d_df = df.groupby(['y', 'z']).agg({'x': 'sum'})
z_min, z_max = df['z'].min(), df['z'].max()
y_min, y_max = df['y'].min(), df['y'].max()
map2d_image = np.ones((z_max + 1, y_max + 1))
map2d_df.reset_index(inplace=True)
map2d_image[(map2d_df['z'].values, map2d_df['y'].values)] = 0
threhold = int(y_min * 0.3 + y_max * 0.7)
map2d_image[:, :threhold] = 0
label_arr = skimage.measure.label(map2d_image, connectivity=2)
index = label_arr.nonzero()
sparse_df = pd.DataFrame({'y': index[1],
                          'z': index[0],
                          'c': label_arr[index]})
cluster_df = sparse_df.groupby('c').agg({'c': ['count']})
cluster_df.columns = ['%s.%s' % e for e in cluster_df.columns.tolist()]
cluster_df.sort_values('c.count', ascending=False, inplace=True)
cluster_df.reset_index(inplace=True)
cluster_df.rename(columns={'index': 'c'})
cluster_df = cluster_df[cluster_df['c.count'] > 10]
cluster_df = cluster_df.tail(len(cluster_df)-1)

sparse_df = sparse_df[sparse_df['c'].isin(cluster_df['c'].values)]
#print(sparse_df)
temp_2d = np.zeros((z_max + 1, y_max + 1))
temp_2d[(sparse_df['z'].values, sparse_df['y'].values)] = 1
import matplotlib.pyplot as plt
plt.imshow(temp_2d)
plt.show()
# 遍历.


# 2d group by outer drop 选取 not NA

