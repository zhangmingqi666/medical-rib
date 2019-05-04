"""
Verify that all ribs_obtain can match with several bounding box
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches
from preprocessing.separated.ribs_obtain import util

main_path = "/Users/jiangyy/projects/medical-rib"
date_tag = ""
ribs_cache_df_path = "{}/data/ribs_df_cache".format(main_path)

"""
unavail_files = pd.read_csv("{}/tools/problems_for_labels/unavail_box{}.csv".format(main_path, date_tag))
unavail_files['id'] = unavail_files['location_id'].apply(lambda x: '-'.join(x.split('-')[0:-1]))
files = unavail_files['id'].unique()
"""

#double_ribs = pd.read_csv("/Users/jiangyy/projects/medical-rib/tools/problems_for_labels/share_box_ribs.csv")
#files = double_ribs['id'].unique()
id_df = pd.read_csv("/Users/jiangyy/projects/medical-rib/tools/problems_for_labels/all_id.csv", dtype={'id': np.str})
files = id_df['id'].unique()

# id,location_id,x.max,x.min,y.max,y.min,z.max,z.min
nii_loc_df_path = "{}/data/csv_files/nii_loc_df.csv".format(main_path)
nii_loc_df = pd.read_csv(nii_loc_df_path, dtype={'id': np.str, 'location_id': np.str})


for _id in files:
    ribs_df_path = "{}/{}.csv".format(ribs_cache_df_path, _id)

    if not os.path.exists(ribs_df_path):
        print("{} ct not exist".format(_id))
        continue

    ribs_df = pd.read_csv(ribs_df_path, dtype={'x': np.int, 'y': np.int, 'z': np.int,
                                                                         'c': np.str})

    rib_data = util.sparse_df_to_arr(arr_expected_shape=[ribs_df['z'].max()+30, ribs_df['x'].max()+30,
                                                         ribs_df['y'].max()+30],
                                     sparse_df=ribs_df, fill_bool=True)

    #related_locations = unavail_files[unavail_files['id'] == _id]['location_id'].unique()
    #temp_df = nii_loc_df[nii_loc_df['location_id'].isin(related_locations)]
    temp_df = nii_loc_df[nii_loc_df['id'] == _id]

    fig, ax = plt.subplots(1)
    ax.imshow(rib_data.sum(axis=1))

    for index, row in temp_df.iterrows():
        # exchange x with y, label reason.
        location_id = row['location_id']
        location_part = location_id.split('-')
        y_min, y_max = row['box.y.min'], row['box.y.max']
        z_min, z_max = row['box.z.min'], row['box.z.max']
        rect = patches.Rectangle((y_min, z_min), (y_max - y_min), (z_max - z_min), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(y_max, z_max, location_part[-1], fontsize=10)

    fig.savefig('{}/tools/results_for_problems/all_yoz_match/{}.png'.format(main_path, _id))
    plt.close(fig)

