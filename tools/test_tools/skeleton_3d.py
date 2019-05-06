import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, skeletonize_3d
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def sparse_df_to_arr(arr_expected_shape=None, sparse_df=None, fill_bool=True):
    """
    :param arr_expected_shape: arr's expected shape
    :param sparse_df:
    :return: expected_arr
    """
    expected_arr = np.zeros(arr_expected_shape)
    point_index = sparse_df['z'].values, sparse_df['x'].values, sparse_df['y'].values
    if fill_bool:
        expected_arr[point_index] = 1
    else:
        expected_arr[point_index] = sparse_df['c']

    del point_index
    del sparse_df
    # gc.collect()
    return expected_arr

df = pd.read_csv("/Users/jiangyy/projects/medical-rib/experiments/logs/135402000611988/label_64241_collect_IS_MULT_RIB.csv")
image_3d = sparse_df_to_arr([df['z'].max() + 10, df['x'].max() + 10, df['y'].max() + 10], df, fill_bool=True)
#data = binary_blobs(200, blob_size_fraction=.2, volume_fraction=.35, seed=1)

skeleton3d = skeletonize_3d(image_3d)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def arr_to_sparse_df_only(binary_arr=None):
    index = binary_arr.nonzero()
    sparse_df = pd.DataFrame({'x': index[1],
                              'y': index[2],
                              'z': index[0],
                              'v': binary_arr[index]})
    del index
    return sparse_df


skeleton3d_df = arr_to_sparse_df_only(binary_arr=skeleton3d)
ax.scatter(skeleton3d_df['x'], skeleton3d_df['y'], skeleton3d_df['z'])
plt.show()