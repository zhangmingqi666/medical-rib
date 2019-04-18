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


print(excel_df[excel_df['cnt']>1])
exit(1)
box_df = pd.read_csv("/Users/jiangyy/projects/medical-rib/preprocessing/prepare_data/multi_box.csv", dtype={'cnt': np.int})

heheh_df = excel_df.merge(box_df, how='left', on='location_id')

heheh_df['cnt'] = heheh_df['cnt'].fillna(1)

heheh_df[['location_id', 'cnt']].to_csv("/Users/jiangyy/Desktop/ccccccc.csv")