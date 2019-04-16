
import pandas as pd
ignore_xml_df = pd.read_csv("./experiments/label_data/ignore.csv")
big_box_df = pd.read_csv("./experiments/label_data/big_box.csv")
label_df = ignore_xml_df.append(big_box_df)
label_df['dataSet_id'] = label_df['xml_name'].apply(lambda x: x[:-4])

print("dataSet number :{}".format(len(label_df)))
join_label_df = pd.read_csv("./data/csv_files/join_label.csv", usecols=['location_id', 'dataSet_id'])

object_label_df = label_df.merge(join_label_df, on='dataSet_id', how='left')
print("label number :{}".format(len(object_label_df)))
object_label_df.to_csv('./experiments/label_data/object_label.csv', index=False)