#coding=utf-8

import pickle
#ribData = RibDataFrame().readDicom(path='/home/jiangyy/projects/medical-rib/dataset/A1077131')

#ribData = pickle.load(open('/home/jiangyy/projects/medical-rib/cache/A1076956.pkl', 'rb'))
ribData = pickle.load(open('/home/jiangyy/projects/medical-rib/cache/135402000573834.pkl', 'rb'))

#self.label[self.label<threholds] = 0
#                self.label[self.label>0] = 1
"""
datacluster1 = dataCluster(data=ribData)
datacluster1.SkimageLabel(threholds=100, connectivity=2,kernel_dim_len=5)
cluster = datacluster1.getTopNCluster(n=100)
c1 = cluster['c'].values[0]
datacluster1.plot3D(c=c1)
"""
#print(datacluster1.getTopNCluster(n=10)[['v', 'v.count', 'x.mean', 'y.mean', 'z.mean']])

#ribData[ribData<400] = 0
#ribData[ribData>400] = 1


#ribData = img_erosion(ribData,kernel_dim_len=3)
#plot_3d(ribData,threshold=0.5)

from bayeslocation.src import remove_rib
print(ribData.shape)
remove = remove_rib.remove(ribData)
result = remove.separate_Bones()
