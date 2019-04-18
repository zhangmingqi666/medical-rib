#coding=utf-8
import pandas as pd
import numpy as np
import os
import argparse
import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def read_excel(excel_path=None):
    """read (patient_id, location_id, rib_type) from **.xls"""
    df = pd.read_excel(excel_path, dtype={'id': np.str, 'location_id': np.str, 'type': np.str},
                       na_values=['nan', 'NaN'])
    df = df[['id', 'location_id', 'type']]
    df = df.fillna(method='ffill')
    return df

excel_df = read_excel("/Users/jiangyy/projects/medical-rib/data/csv_files/rib_type_location.xls")
excel_df.drop(columns=['id'], inplace=True)


for idx, row in excel_df.iterrows():
    nii_path = "/Users/jiangyy/projects/medical-rib/data/nii_csv_files/{}.csv".format(row['location_id'])

    if not os.path.exists(nii_path):
        print("{} not exist".format(row['location_id']))
        continue
    point_df = pd.read_csv(nii_path)

    plt.figure(figsize=(12, 4))
    grid = plt.GridSpec(4, 12, wspace=0.5, hspace=0.5)
    plt.subplot(grid[0:4, 0:4])
    plt.scatter(point_df['x'], point_df['y'], s=20, color='green')

    plt.subplot(grid[0:4, 4:8])
    plt.scatter(point_df['y'], point_df['z'], s=20, color='green')

    plt.subplot(grid[0:4, 8:12])
    plt.scatter(point_df['z'], point_df['x'], s=20, color='green')
    plt.savefig("/Users/jiangyy/Desktop/point_plto/{}.png".format(row['location_id']))
    plt.show(False)

    #plt.show()

    #cnt = input("cnt input:")
    #print("#####,{},{},{}".format(idx, row['location_id'], cnt))
