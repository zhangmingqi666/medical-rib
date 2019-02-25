
import os
import pickle as pkl
import pandas as pd


def get_df_id_shape(pkl_path):
    files = os.listdir(pkl_path)
    id_list = []
    id_pkl_shape_list = []
    for file in files:
        f_path = os.path.join(pkl_path, file)
        rib_data = pkl.load(open(f_path, 'rb'))
        id_list.append(file)
        id_pkl_shape_list.append(rib_data.shape)
    return pd.DataFrame({'id': id_list, 'pkl_list': id_pkl_shape_list})


df_pkl = get_df_id_shape("/Users/jiangyy/DataSet/rib_dataSet/updated48labeled_1.31/pkl_cache")
df_pkl_bak = get_df_id_shape("/Users/jiangyy/DataSet/rib_dataSet/updated48labeled_1.31/pkl_cache_bak")

df = df_pkl.merge(df_pkl_bak, how='outer', on='id')
print(df)
print(df.columns)
