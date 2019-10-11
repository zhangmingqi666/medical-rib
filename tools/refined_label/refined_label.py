import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
import numpy as np
import xml.etree.ElementTree as ET
import os

csv_path = '/Users/jiangyy/projects/medical-rib/data/csv_files'
voc_path = '/Users/jiangyy/projects/medical-rib/data/voc2007'
join_label_df = pd.read_csv('{}/join_label.csv'.format(csv_path))
# offset_df = pd.read_csv('{}/offset.csv'.format(csv_path), names=['dataSet_id', 'offset-x', 'offset-y', 'offset-z'], header=None)
png_path = '{}/JPEGImages'.format(voc_path)
png_name_df = pd.DataFrame({'name': os.listdir(png_path)})
png_name_df['dataSet_id'] = png_name_df['name'].apply(lambda x: '-'.join(x.split('-')[0:-1]))
png_name_df['name'] = png_name_df['name'].apply(lambda x: '{}/{}'.format(png_path, x))

local_location_df = pd.read_csv('{}/after_tighten.csv'.format(csv_path),)

df = png_name_df.merge(join_label_df, on=['dataSet_id']).merge(local_location_df, on=['id', 'location_id'])

df['id,location_id,box.x.max,box.x.min,box.y.max,box.y.min,box.z.max,box.z.min'.split(',')].to_csv('{}/resort_aftertighten.csv'.format(csv_path))

color_list = ['r',  'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r',  'g', 'b', 'c', 'm', 'y', 'k', 'w']

for _, row in png_name_df[100:].iterrows():

    plt.figure(figsize=(4, 4))
    grid = plt.GridSpec(4, 4, wspace=0.5, hspace=0.5)
    plt.subplot(grid[0:4, 0:4])
    print(row['dataSet_id'])
    image = Image.open(row['name'])
    plt.imshow(image)

    temp_local_location_df = df[df['dataSet_id'] == row['dataSet_id']].copy(deep=True).reset_index(drop=True)
    for _id, _row in temp_local_location_df.iterrows():
        truth = [_row['box.y.min'], _row['box.x.min'], _row['box.y.max'], _row['box.x.max']]
        _label = _row['location_id'].split('-')[-1]
        plt.gca().add_patch(patches.Rectangle((truth[0], truth[1]), (truth[2] - truth[0]), (truth[3] - truth[1]),
                                              linewidth=1, edgecolor=color_list[_id], facecolor='none', label=_label))

    ax = plt.gca()
    ax.invert_yaxis()
    ax.legend()
    plt.show()
