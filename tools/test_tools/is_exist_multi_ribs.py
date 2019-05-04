
import os
import pandas as pd
import numpy as np
ribs_cache_folder = "./data/ribs_df_cache"

for file in os.listdir(ribs_cache_folder):
    ct_id = file[0:-4]
    rib_data_df_path = os.path.join(ribs_cache_folder, file)
    if not(os.path.exists(rib_data_df_path) and rib_data_df_path.endswith(".csv")):
        print("{} unavailable".format(ct_id))

    data_df = pd.read_csv(rib_data_df_path, dtype={'x': np.int, 'y': np.int, 'z': np.int, 'c': np.str})
