import os
import pandas as pd

path="/data/jiangyy/rib_dataSet"
excel_path="rib_type_location.xls"
files = ['updated48labeled_1.31', 'added51-100labeled_2.21', 'added101-150labeled_2.21', 'added151-200labeled_2.21', 'added201-237labeled_2.21']

label_id = []
label_nii_id = []
label_group = []

dataset_id = []
dataset_group = []

for file in files:
    data_file = os.path.join(path, file)
    dataset_file = os.path.join(data_file, "dataset")
    label_file = os.path.join(data_file, "label")

    for e in os.listdir(dataset_file):
        dataset_id.append(e)
        dataset_group.append(file)

    for f in os.listdir(label_file):
        if f.startswith('.'):
            continue
        temp = os.path.join(label_file, f)
        for g in os.listdir(temp):
            label_id.append(f)
            label_nii_id.append(g)
            label_group.append(file)

label_df = pd.DataFrame({'id': label_id, 'nii_id': label_nii_id, 'label_group': label_group})
dataset_df = pd.DataFrame({'id': dataset_id, 'dataset_group': dataset_group})
data_df = label_df.merge(dataset_df, how='outer', on='id')

excel_df = pd.read_excel(excel_path)
excel_df = excel_df[['id', 'loaction_id', 'type']]
excel_df = excel_df.fillna(method='ffill')
excel_df.rename(columns={'loaction_id': 'nii_id'}, inplace=True)
excel_df['type'] = 'From xls'
df = excel_df.merge(data_df, how='outer', on=['id', 'nii_id'])
df.to_csv('hehe.csv', index=False)
