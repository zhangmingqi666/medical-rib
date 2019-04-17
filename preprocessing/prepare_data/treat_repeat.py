
import pandas as pd

repeat_df = pd.read_csv("/Users/jiangyy/projects/medical-rib/data/csv_files/join_label.csv")
hist_repeat_df = repeat_df.groupby('dataSet_id').agg({'dataSet_id': 'count'})
hist_repeat_df.columns = ['count']
hist_repeat_df.reset_index(inplace=True)
hist_repeat_df = hist_repeat_df[hist_repeat_df['count']>1]
repeat_label = hist_repeat_df.merge(repeat_df, on='dataSet_id', how='left')

print(repeat_label[['dataSet_id', 'location_id']])

#for _, row in hist_repeat_df.iterrows():
    #print('dataSet_id', row['135402000151454-2280'], 'count', row['count'])

    #for _, box_row in