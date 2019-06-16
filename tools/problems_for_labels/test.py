import pandas as pd

df1 = pd.read_csv('incomplete.csv', delimiter=':', sep=':')
df2 = pd.read_csv('unavail_box.csv')
df2['id'] = df2['location_id'].apply(lambda x: '-'.join(x.split('-')[0:-1]))

df3 = df2.merge(df1, on='id', how='left')
df4 = df3[df3['tag'].notnull()]
print(df4)
#print(df4['id'].unique())