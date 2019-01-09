import numpy as np
import pandas as pd
from pandas import DataFrame

def df2array(df,columns,values,shape):

    temp_df = self.getDatabyCluster(c=c)
    tempdata = np.zeros(self.Data.shape)
    pointIndex = temp_df['z'].values, temp_df['x'].values, temp_df['y'].values
    tempdata[pointIndex] = temp_df['v']
    tempdata[tempdata>0] = 1
