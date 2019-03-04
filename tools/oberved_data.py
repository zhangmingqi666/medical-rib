
import pandas as pd
RIB_DF_CACHE_PATH="~/Desktop/logs/135402000150175/label_77051_collect_NOT_RIB.csv"
bone_df = pd.read_csv(RIB_DF_CACHE_PATH)
print(bone_df.head(10))
z_max, x_max, y_max = bone_df['z'].max() + 1, bone_df['x'].max()+1, bone_df['y'].max()+1

from preprocessing.separated.ribs_obtain import util

arr = util.sparse_df_to_arr([z_max, x_max, y_max], sparse_df=bone_df, fill_bool=False)
print(arr.shape)
util.plot_3d(arr, threshold=150)