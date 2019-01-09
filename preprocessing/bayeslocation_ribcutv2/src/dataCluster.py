#coding=utf-8
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
import numpy as np
import pandas as pd
from bayeslocation.src.plot import plot_3d
import skimage.morphology as sm
import matplotlib.pyplot as plt

#erosion
def img_erosion(img, kernel_dim_len=5):
    l = kernel_dim_len
    kernel = np.zeros([l,l,l],dtype=np.uint8)
    return sm.binary_opening(img,sm.ball(1))

#dilation
def img_dilation(img, kernel_dim_len=3):
    l = kernel_dim_len
    kernel = np.ones([l,l,l],dype=np.uint8)
    return sm.dilation(img,kernel)

class ClusterBone:
    def __init__(self):
        pass


class dataCluster:
    def __init__(self, data=None):
        self.Data = data
        self.df = None
        self.Cluster = None
        #pass

    def SkimageLabel(self, threholds=400, connectivity=2, kernel_dim_len=3):
        self.label = self.Data.copy()
        self.label[self.label<threholds] = 0
        self.label[self.label>0] = 1
        #self.label = img_erosion(self.label,kernel_dim_len)
        self.label = skimage.measure.label(self.label, connectivity=connectivity)
        index_label = self.label.nonzero()
        self.df = pd.DataFrame({'x': index_label[1],
                                'y': index_label[2],
                                'z': index_label[0],
                                'c': self.label[index_label],
                                'v': self.Data[index_label]})
        # Maybe Cluster can be an independent class.
        self.Cluster = self.df.groupby('c').agg({'x': ['mean', 'min', 'max'], 'y': ['mean', 'min', 'max'],
                                                 'z': ['mean', 'min', 'max'], 'c': ['count']})
        self.Cluster.columns = ['%s.%s' % e for e in self.Cluster.columns.tolist()]
        self.Cluster['x.length'] = self.Cluster['x.max'] - self.Cluster['x.min'] + 1
        self.Cluster['y.length'] = self.Cluster['y.max'] - self.Cluster['y.min'] + 1
        self.Cluster['z.length'] = self.Cluster['z.max'] - self.Cluster['z.min'] + 1
        self.Cluster.sort_values('c.count', ascending=False, inplace=True)
        self.Cluster.reset_index(inplace=True)
        self.Cluster.rename(columns={'index': 'c'}, inplace=True)

    def getTopNCluster(self, n=None):
        if n is None:
            return self.Cluster
        return self.Cluster.head(n)

    def getDatabyCluster(self, c=0):
        return self.df[self.df['c'] == c]

    def getDataShapeByCluster(self, c=0):
        temp_df = self.getDatabyCluster(c=c)
        return temp_df[['z.length', 'x.length', 'y.length']].values[0] + 1

    def getDataMinIndexByCluster(self, c=0):
        temp_df = self.getDatabyCluster(c=c)
        return temp_df[['z.min', 'x.min', 'y.min']].values[0]

    def getDataMaxIndexByCluster(self, c=0):
        temp_df = self.getDatabyCluster(c=c)
        return temp_df[['z.max', 'x.max', 'y.max']].values[0]

    def plot3D(self, c=0, threshold=0.5):
        temp_df = self.getDatabyCluster(c=c)
        tempdata = np.zeros(self.Data.shape)
        pointIndex = temp_df['z'].values, temp_df['x'].values, temp_df['y'].values
        tempdata[pointIndex] = temp_df['v']
        tempdata[tempdata>0] = 1
        print(np.max(tempdata))
        print(np.min(tempdata))
        tempdata = skimage.morphology.skeletonize_3d(tempdata)

        plot_3d(tempdata, threshold=threshold)

        #print()




