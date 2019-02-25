
"""
Verify that all ribs can match with several bounding box
"""


import os
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches
pkl_path = "/Users/jiangyy/DataSet/rib_dataSet/updated48labeled_1.31/pkl_cache"
files = os.listdir(pkl_path)

# id,location_id,x.max,x.min,y.max,y.min,z.max,z.min
nii_loc_df_path = "/Users/jiangyy/DataSet/rib_dataSet/nii_loc_df.csv"
df = pd.read_csv(nii_loc_df_path, dtype={'id': np.str,'location_id': np.str})

for file in files:
    f_path = os.path.join(pkl_path, file)
    rib_data = pkl.load(open(f_path, 'rb'))
    rib_data[rib_data <= 400] = 0
    rib_data[rib_data > 400] = 1
    save_f_name = file.replace('.pkl', '.png')
    temp_id = file.replace('.pkl', '')
    temp_df = df[df['id'] == temp_id]

    fig, ax = plt.subplots(1)
    ax.imshow(rib_data.sum(axis=1))

    for index, row in temp_df.iterrows():
        # exchange x with y, label reason.
        y_min, y_max = row['y.min'], row['y.max']
        z_min, z_max = row['z.min'], row['z.max']
        rect = patches.Rectangle((y_min, z_min), (y_max - y_min), (z_max - z_min), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.savefig(os.path.join('/Users/jiangyy/projects/medical-rib/tools/Verify_logs/Verify_data_join_label', save_f_name))

