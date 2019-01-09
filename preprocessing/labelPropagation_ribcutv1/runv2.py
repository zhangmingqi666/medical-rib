
import numpy as np
import pickle
m = pickle.load(open("tmp2.txt", "rb"))
m[m<400] = 0
aaa = m.nonzero()
#print(len(aaa[0]))
data = np.concatenate((aaa[0].reshape(-1,1),aaa[1].reshape(-1,1)),axis=1)
"""
from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=20, random_state=3).fit_predict(data)
import matplotlib.pyplot as plt
plt.scatter(data[:, 0], data[:, 1], c=y_pred)
plt.show()
"""

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=24, affinity='euclidean', linkage='single')
cluster.fit_predict(data)
import matplotlib.pyplot as plt
plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()
#list = []
#labelunique = np.unique(cluster.labels_)
#for e in labelunique
import pandas as pd
df = pd.DataFrame({'x':data[:,0],'z':data[:,1],'label':cluster.labels_})
average = df.groupby('label').agg({'x':'mean','z':'mean'})
average.reset_index(inplace=True)
average.columns = ['label','x_mean','z_mean']
average['x_mean'] = average['x_mean'].apply(np.int)
average['z_mean'] = average['z_mean'].apply(np.int)
average['y_mean'] = 400
print(average)



#print(cluster.labels_)