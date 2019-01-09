#coding=utf-8
path="/Users/jiangyy/DataSet/rib_dataSet/first20labeled/rib_df_cache"
import os
import pandas as pd
#for e in os.listdir(path):
#    f = os.path.join(path, e)
#
#    df = pd.read_csv(f)
#    df.rename(columns={'id': 'c'}, inplace=True)
#    df.to_csv(f, index=False）

in_path = "/Users/jiangyy/Desktop/rib_df_input_dataframe"
out_path = "/Users/jiangyy/Desktop/merge"


def merge_all_ribs_for_person(folder='/Users/jiangyy/Desktop/rib_df_input_dataframe/id'):
    df = pd.DataFrame({})
    files = os.listdir(folder)
    for e in files:
        if not e.endswith('.csv'):
            continue
        temp_df = pd.read_csv(os.path.join(folder, e), index_col=0)
        temp_df['c'] = e.replace('.csv', '')
        df = df.append(temp_df)
    return df


for f in os.listdir(in_path):
    if f.startswith("."):
        continue
    person_path = os.path.join(in_path, f)
    temp_df = merge_all_ribs_for_person(person_path)
    temp_df.to_csv(os.path.join(out_path,"{}.csv".format(f)), index=False)