
import pandas as pd
ignore_xml_df = pd.read_csv("./experiments/label_data/ignore.csv")
big_box_df = pd.read_csv("./experiments/label_data/big_box.csv")
label_df = ignore_xml_df.append(big_box_df)
label_df['dataSet_id'] = label_df['xml_name'].apply(lambda x: x[:-4])

print("dataSet number :{}".format(len(label_df)))
join_label_df = pd.read_csv("./data/csv_files/join_label.csv")


object_label_df = label_df.merge(join_label_df, on='dataSet_id', how='left')

for idx, row in object_label_df.iterrows():
    dataset_id, location_id = row['dataSet_id'], row['location_id']

    png_path = "/Users/jiangyy/voc2007.xoy/JPEGImages/{}.jpg"

    nii_csv_path
