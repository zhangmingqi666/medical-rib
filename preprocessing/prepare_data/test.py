

import pandas as pd
import numpy as np
csv_dataset_folder = '/Users/jiangyy/projects/medical-rib/data/ribs_df_cache'
ct_id = '135402000151454'
data_df = pd.read_csv("{}/{}.csv".format(csv_dataset_folder, ct_id), dtype={'x': np.int,
                                                                            'y': np.int,
                                                                            'z': np.int,
                                                                            'c': np.str})
print(data_df['c'].value_counts())
# get global erea for every ribs.
range_data_df = data_df.groupby('c').agg({'x': ['min', 'max'],
                                          'y': ['min', 'max'],
                                          'z': ['min', 'max']})

print(range_data_df)
#range_data_df.to_csv("./data/temp/range1-{}.csv".format(ct_id), index=True)
