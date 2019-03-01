
import pandas as pd
patient_id=''
class_id=''
RIB_DF_CACHE_PATH="./data/ribs_df_cache/{}.csv".format(patient_id)
df = pd.read_csv(RIB_DF_CACHE_PATH)
bone_df = df[df['class_id'] == class_id]
z_max, x_max, y_max = bone_df['z'].max() + 1, bone_df['x'].max()+1, bone_df['y'].max()+1

from preprocessing.separated.ribs_obtain import util

arr = util.sparse_df_to_arr([z_max, x_max, y_max], sparse_df=bone_df, fill_bool=True)
util.plot_3d(arr)