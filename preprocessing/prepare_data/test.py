#coding=utf-8
import pandas as pd
import numpy as np
import os
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')


def read_excel(excel_path=None):
    """read (patient_id, location_id, rib_type) from **.xls"""
    df = pd.read_excel(excel_path, dtype={'id': np.str, 'location_id': np.str, 'type': np.str, 'cnt':np.int},
                       na_values=['nan', 'NaN', np.NAN, np.nan])
    df = df[['id', 'location_id', 'type', 'cnt']]
    df['id'] = df['id'].replace('nan', np.NAN)
    df = df.fillna(method='ffill', axis=0)
    return df


excel_df = read_excel("/Users/jiangyy/projects/medical-rib/data/csv_files/rib_type_location.xls")
print(len(excel_df))
location_df = pd.read_csv("/Users/jiangyy/projects/medical-rib/data/csv_files/nii_loc_df.csv")
location_df_cnt = location_df.groupby(['id', 'location_id']).agg({'location_id': ['count']})
location_df_cnt.columns = ['box.count']
location_df_cnt.reset_index(inplace=True)

print(len(location_df_cnt))

ddd = excel_df.merge(location_df_cnt, on=['id', 'location_id'])

print(ddd[(ddd['cnt'] < ddd['box.count']) & (ddd['cnt'] != 1)])




