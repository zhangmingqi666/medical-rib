import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
import numpy as np
import xml.etree.ElementTree as ET
import os

csv_path = '/Users/jiangyy/projects/medical-rib/data/csv_files'
join_label_df = pd.read_csv('{}/join_label.csv'.format(csv_path))
offset_df = pd.read_csv('{}/offset.csv'.format(csv_path), names=['dataSet_id', 'range.x.min', 'range.y.min', 'range.z.min'])
refined_df = pd.read_csv('{}/resort_aftertighten.csv'.format(csv_path))
refined_df = refined_df[refined_df['comment'].isnull()]
df = refined_df.merge(join_label_df, on='location_id').merge(offset_df, on='dataSet_id')
print(df.columns)

for e in ['x', 'y', 'z']:
    df['box.{}.max'.format(e)] = df.apply(lambda row:row['box.{}.max'.format(e)]+row['range.{}.min'.format(e)], axis=1)
    df['box.{}.min'.format(e)] = df.apply(lambda row:row['box.{}.min'.format(e)]+row['range.{}.min'.format(e)], axis=1)

df = df['location_id,box.x.max,box.x.min,box.y.max,box.y.min,box.z.max,box.z.min'.split(',')]
df.to_csv('{}/restore_glb.csv'.format(csv_path), index=False)
