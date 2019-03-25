
import pandas as pd
from preprocessing.separated.ribs_obtain.util import sparse_df_to_arr, plot_binary_array

ids = ['135402000404222', '135402000404891', '135402000404065', '135402000357765', '135402000555091',
       '135402000572309', '135402000555684', '135402000404090']

main_path = '../experiments/last_logs'
output_main_path = '../experiments/debug_logs/Verify_observe_data'
for id in ids:
    for file in ['local.csv', 'bone_data.csv']:
        csv_path = "{}/{}/{}".format(main_path, id, file)
        bone_df = pd.read_csv(csv_path)
        z_max, x_max, y_max = bone_df['z'].max() + 1, bone_df['x'].max()+1, bone_df['y'].max()+1
        image = sparse_df_to_arr([z_max, x_max, y_max], sparse_df=bone_df, fill_bool=False)
        plot_binary_array(image, title=file.replace('csv', ''), save=True,
                          save_path='{}/{}_{}.png'.format(output_main_path, id, file.replace('csv', '')))

