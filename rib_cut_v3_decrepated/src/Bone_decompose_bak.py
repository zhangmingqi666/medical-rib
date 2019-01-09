import numpy as np
import pandas as pd
import time
import skimage.morphology as sm
import skimage
import matplotlib.pyplot as plt
import gc
from skimage.measure import label
from .util import *






class remove:
    def __init__(self, array=None):
        self.array = array
        self.df = self.array2df(self.array)
        self.shape = self.array.shape
        self.LargestBone = None
        self.x_lenght_large_bone = None
        self.y_lenght_large_bone = None
        self.BoneList = pd.DataFrame({})

    def getData(self,data):
        self.df = data

        #array转dataframe
    def array2df(self,array):
        index_z, index_x, index_y = array.nonzero()
        df = pd.DataFrame({'x':index_x,'y':index_y, 'z':index_z, 'v':array[index_z,index_x,index_y]})
        return df

    def separate_bones(self, hu_threshold=400):
        """
        :param hu_threshold:
        :return:
        """

        #store all separate point

        label_arr = self.array.copy()
        label_arr[label_arr < hu_threshold] = 0
        label_arr[label_arr >= hu_threshold] = 1
        labelarr = skimage.measure.label(label_arr, connectivity=2)

        temp_df, cluster_df = arr_to_sparse_df(label_arr=labelarr, pixel_arr=self.array, sort=True, sort_key='c.count',
                                               keep_by_threshold=True, threshold_min=4500)

        iter = 0

        debug_temp_list = []
        Rib_Classes = []

        while (MaxBoneIsSpine(temp_df, cluster_df, zshape=self.shape[0], yshape=self.shape[2], xshape=self.shape[1]) and
                SpineConnectedRib(temp_df, cluster_df, zshape=self.shape[0], yshape=self.shape[2], xshape=self.shape[1])):
            #self.BoneList.append(cluster_df[1:])
            #self.BoneList['c'] = self.BoneList['c'].apply(lambda x: x+10000000)

            time0 = time.time()
            iter += 1
            print('Iteration %d:'%iter)
            MaxCluster_ID = cluster_df['c'][0]


            # choose top 1
            time1 = time.time()
            label_arr = sparse_df_to_arr(arr_expected_shape=self.shape, sparse_df=temp_df[temp_df['c'] == MaxCluster_ID])
            label_arr[label_arr > 0] = 1
            """
            print(len(labelarr.nonzero()[0]))
            
            plt.imshow(labelarr.sum(axis=1))
            plt.show()
            """
            print('Extracted the maximum bone, elapsed time is %d seconds'%(time.time()-time1))
            time1 = time.time()

            labelarr = loop_morphology_binary_opening(labelarr, use_cv=False, transfer_to_binary=False, printinfo=False)

            """
            # opening
            r = 1
            #while True:
            temp_nozero_sum = labelarr.sum()
            while True:
                labelarr = sm.binary_opening(labelarr, sm.ball(r))
                if labelarr.sum() != temp_nozero_sum:
                    break
                r = r+0.4
                print('erosion:', r)
            labelarr[labelarr==True] = 1
            labelarr[labelarr==False] = 0
            """

            print('Opening Calculation, elapsed time is %d seconds'%(time.time()-time1))
            time1 = time.time()
            # label
            #exit(1)
            labelarr = skimage.measure.label(labelarr, connectivity=2)

            print('Max Connected bone labeled, elapsed time is %d seconds'%(time.time()-time1))
            time1 = time.time()

            """
            cluster, sort
            """
            temp_df, cluster_df = arr_to_sparse_df(label_arr=labelarr, pixel_arr=self.array, sort=True,
                                                   sort_key='c.count',
                                                   keep_by_threshold=True, threshold_min=4500)

            print('cluster,sort,delete least, elapsed time is %d seconds'%(time.time()-time1))
            time1 = time.time()





            print('Find ribs and its joint point, elapsed time is %d seconds'%(time.time()-time1))

        del labelarr
        gc.collect()
        # return the DataFrame that only include spine
        return temp_df[temp_df['c'] == cluster_df['c'][0]]

    def get_rib_by_cut_spine(self):
        spine_df = self.separate_Bones()
        spine_arr = DataFrame2Arr(self.shape, spine_df)
        plt.imshow(spine_arr.sum(axis=1))
        plt.show()
        del spine_df
        gc.collect()

        # closing operation
        for i in range(6):
            spine_arr = sm.binary_dilation(spine_arr, selem=sm.ball(3))
        plt.imshow(spine_arr.sum(axis=1))
        plt.show()
        self.array[self.array < 400] = 0
        self.array[self.array >= 400] = 1
        self.array[spine_arr > 0] = 0
        """
        #self.array = skimage.measure.label(self.array, connectivity=2)
        #temp_df, cluster_df = LabelArrPixel2DataFrame(self.array, self.array, sort=True,keepThresholds=True, Threholds_Min=4500)
        print(3)
        rib_class = []
        for index, e in cluster_df[1:].iterrows():
            # print('hello world:',e['c'])
            singlebone = Bone(bone_data=temp_df[temp_df['c'] == e['c']], shape=label_arr.shape, naturally_shedding=True)
            singlebone.set_bone_type()
            singlebone.print_bone_info()
            if singlebone.is_rib():
                rib_class.append(e['c'])
        print(4)
        plot_separater_bone(temp_df[temp_df['c'] == cluster_df['c'][0]], temp_df[temp_df['c'].isin(rib_class)],self.shape)
        """
        plt.imshow(self.array.sum(axis=1))
        plt.show()
        # del temp_df
        del spine_arr
        gc.collect()
