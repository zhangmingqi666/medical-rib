

RIB_DF_CACHE_PATH="~/Desktop/logs/135402000150175/label_77051_collect_NOT_RIB.csv"
import pandas as pd

df = pd.read_csv(RIB_DF_CACHE_PATH)

print(len(df))
grpby = df.groupby('z').agg({'y':['mean']})
grpby.columns = ['y.mean']
print(grpby[grpby['y.mean'] > 200])

ccc = grpby[grpby['y.mean'] > 200]
ccc.reset_index(inplace=True)
ccc.rename(columns={'index': 'z'})
print(type(ccc['z']))

df_in = df[df['z'].isin(ccc['z'])]

print(df_in['z'].value_counts())