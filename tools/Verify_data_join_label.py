"""
Verify that all ribs_obtain can match with several bounding box
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches


pkl_path = "../data/ribs_df_cache"
files = os.listdir(pkl_path)

# id,location_id,x.max,x.min,y.max,y.min,z.max,z.min
nii_loc_df_path = "../data/csv_files/merge_nii_df.csv"
df = pd.read_csv(nii_loc_df_path, dtype={'id': np.str, 'location_id': np.str})
from preprocessing.separated.ribs_obtain import util
for file in files:
    f_path = os.path.join(pkl_path, file)
    df1 = pd.read_csv(f_path)
    rib_data = util.sparse_df_to_arr(arr_expected_shape=[df1['z'].max()+30, df1['x'].max()+30, df1['y'].max()+30],
                                     sparse_df=df1, fill_bool=True)
    save_f_name = file.replace('.csv', '.png')
    temp_id = file.replace('.csv', '')
    temp_df = df[df['id'] == temp_id]

    fig, ax = plt.subplots(1)
    ax.imshow(rib_data.sum(axis=1))

    print("hello")
    for index, row in temp_df.iterrows():
        # exchange x with y, label reason.
        location_id = row['location_id']
        location = location_id.split('-')
        location_name = "**" if len(location) == 1 else location[-1]
        y_min, y_max = row['box.y.min'], row['box.y.max']
        z_min, z_max = row['box.z.min'], row['box.z.max']
        rect = patches.Rectangle((y_min, z_min), (y_max - y_min), (z_max - z_min), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(y_max, z_max, location_name, fontsize=10)

    fig.savefig(os.path.join('../experiments/debug_logs/Verify_data_join_label',
                             save_f_name))
    plt.close(fig)

