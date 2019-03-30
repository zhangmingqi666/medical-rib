
import os
import pandas as pd
import numpy as np
path = "/Users/jiangyy/Desktop/bone_info_merges"
update_path = "/Users/jiangyy/Desktop/bone_info_update.csv"
update_df = pd.read_csv(update_path, dtype={'ID': np.str, 'class_id': np.float})

df = pd.DataFrame({})
last_id = ""
for ID in update_df['ID'].unique():

    f = "{}/{}.csv".format(path, ID)
    values = update_df[update_df['ID']==ID]['class'].values
    # print(values)
    #print(f, v)
    temp_df = pd.read_csv(f)
    # print(temp_df)
    temp_df.loc[temp_df['class_id'].isin(values), 'target'] = 2.0
    temp_df.to_csv(os.path.join("/Users/jiangyy/projects/medical-rib/data/bone_info_merges", "{}.csv".format(ID)),
                   index=False)
    df = df.append(temp_df)