import numpy as np
from  bayeslocation.src.Bone import Bone
import skimage.morphology as sm
import skimage
from skimage.measure import label
import pandas as pd
import time
import matplotlib.pyplot as plt
import gc
import objgraph


def plot_separater_bone(top_df, rib_df, shape, scatter_point_list):

    top_groupby_df = top_df.groupby(['y', 'z']).agg({'x': 'count'})
    top_groupby_df.columns = ['x.count']
    top_groupby_df.reset_index(inplace=True)
    print(top_groupby_df.columns)
    top_arr = np.zeros((shape[0], shape[2]))
    top_index = top_groupby_df['z'].values, top_groupby_df['y'].values
    top_arr[top_index] = top_groupby_df['x.count']

    rib_groupby_df = rib_df.groupby(['y', 'z']).agg({'x':'count'})
    rib_groupby_df.columns = ['x.count']
    rib_groupby_df.reset_index(inplace=True)
    rib_arr = np.zeros((shape[0], shape[2]))
    rib_index = rib_groupby_df['z'].values, rib_groupby_df['y'].values
    rib_arr[rib_index] = rib_groupby_df['x.count']

    scatter_point = np.array(scatter_point_list)

    plt.subplot(121)
    plt.imshow(top_arr+rib_arr)
    # plt.imshow(rib_arr)
    plt.scatter(x=scatter_point[:, 2], y=scatter_point[:, 0], c='r')
    plt.subplot(122)
    plt.imshow(top_arr)
    plt.show()


def LabelArrPixel2DataFrame(labelArr, pixelArr, sort=False, sortkey='c.count',
                            keepTopN = False, TopN = 30,
                            keepThresholds = False, Threholds_Min = 2000):
    index = labelArr.nonzero()
    temp_df = pd.DataFrame({'x': index[1],
                            'y': index[2],
                            'z': index[0],
                            'c': labelArr[index],
                            'v': pixelArr[index]})
    cluster_df = temp_df.groupby('c').agg({'x': ['mean', 'min', 'max'],
                                           'y': ['mean', 'min', 'max'],
                                           'z': ['mean', 'min', 'max'],
                                           'c': ['count']})
    cluster_df.columns = ['%s.%s'%e for e in cluster_df.columns.tolist() ]

    if sort:
        cluster_df.sort_values('c.count', ascending=False, inplace=True)
    cluster_df.reset_index(inplace=True)
    cluster_df.rename(columns={'index': 'c'})

    if keepTopN:
        cluster_df=cluster_df.head(TopN)

    if keepThresholds:
        cluster_df = cluster_df[cluster_df['c.count'] > Threholds_Min]

    for e in ['x', 'y', 'z']:
        cluster_df['%s.length'%e] = cluster_df.apply(lambda row: row['%s.max'%e] - row['%s.min'%e], axis=1)

    return temp_df, cluster_df

def DataFrame2Arr(shape, temp_df):
    tempdata = np.zeros(shape)
    point_index = temp_df['z'].values, temp_df['x'].values, temp_df['y'].values
    tempdata[point_index] = temp_df['v']
    tempdata[tempdata > 0] = 1
    del point_index
    del temp_df
    return tempdata

def MaxBoneIsSpine(temp_df, cluster_df, zshape=400, yshape=800, xshape=800):
    if cluster_df['z.length'].values[0] > 0.5 * zshape:
        return True
    return False

def SpineConnectedRib(temp_df, cluster_df, zshape=400, yshape=800, xshape=800):
    if cluster_df['y.length'].values[0] > 0.4 * yshape:
        return True
    return False





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

    def separate_Bones(self, BoneThreshold=400):

        #store all separate point
        all_joint_list = []

        labelarr = self.array.copy()
        labelarr[labelarr < BoneThreshold] = 0
        labelarr[labelarr >= BoneThreshold] = 1
        print(labelarr[labelarr.nonzero()])
        labelarr = skimage.measure.label(labelarr, connectivity=2)

        temp_df, cluster_df = LabelArrPixel2DataFrame(labelarr, self.array, sort=True,
                                                      keepThresholds=True, Threholds_Min=4500)
        print('cluster_df')
        print(cluster_df)


        # collect removed ribs,
        # save the biggest spine and rib , waiting for the joint point to cut
        jointList = []

        iter = 0


        print('second show growth')
        # objgraph.show_growth()
        debug_temp_list = []
        Rib_Classes = []
        # debug_temp_df = pd.DataFrame({})
        for index, e in cluster_df[1:].iterrows():
            objgraph.show_growth()
            # print('hello world:',e['c'])
            singlebone = Bone(bone_data=temp_df[temp_df['c'] == e['c']], shape=labelarr.shape, naturally_shedding=True)
            singlebone.set_bone_type()
            singlebone.print_bone_info()
            if singlebone.is_rib():
                #print("VOLUME:",singlebone.bonevolume)
                #plt.imshow(DataFrame2Arr(shape=self.shape,temp_df=temp_df[temp_df['c'] == e['c']]).sum(axis=1))
                #show_point = np.array(singlebone.get_joint_point())
                #plt.scatter(x=show_point[:,2],y=show_point[:,0],c='r')
                #plt.show()
                #if abs(singlebone.get_joint_point()[0][2] - self.shape[2]/2) < 100:
                debug_temp_list.extend(singlebone.get_joint_point())
                Rib_Classes.append(e['c'])
                #debug_temp_df = debug_temp_df.append(singlebone.get_rawdata().copy(deep=True))
            del singlebone
            gc.collect()
            #objgraph.show_growth()

        plot_separater_bone(temp_df[temp_df['c'] == cluster_df['c'][0]], temp_df[temp_df['c'].isin(Rib_Classes)],self.shape, debug_temp_list)
        all_joint_list.extend(debug_temp_list)
        """
        debug_img_arr = DataFrame2Arr(self.shape, temp_df[temp_df['c'] == cluster_df['c'][0]])
        debug_img_arr_cpy = debug_temp_df
        debug_index = debug_temp_df['z'].values, debug_temp_df['x'].values, debug_temp_df['y'].values
        debug_img_arr[debug_index] = 1

        scatter_point = np.array(debug_temp_list)

        plt.subplot(121)
        plt.imshow(debug_img_arr.sum(axis=1))
        plt.scatter(x=scatter_point[:,2],y=scatter_point[:,0],c='r')
        plt.subplot(122)
        plt.imshow(debug_img_arr_cpy.sum(axis=1))
        plt.show()
        del debug_img_arr_cpy
        gc.collect()
        """

        """
        for i in range(len(cluster_df['c'].unique())):
            img_arr = DataFrame2Arr(self.shape, temp_df[temp_df['c'] == cluster_df['c'][i]])
            plt.imshow(img_arr.sum(axis=1))
            plt.text(800, 200, 'cluster:%d' % i, bbox=dict(facecolor='red', alpha=1))
            plt.show()
        """

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
            labelarr = DataFrame2Arr(self.shape, temp_df[temp_df['c'] == MaxCluster_ID])
            labelarr[labelarr > 0] = 1
            """
            print(len(labelarr.nonzero()[0]))
            
            plt.imshow(labelarr.sum(axis=1))
            plt.show()
            """
            print('Extracted the maximum bone, elapsed time is %d seconds'%(time.time()-time1))
            time1 = time.time()

            # opening
            print(len(labelarr.nonzero()[0]))
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
            print(len(labelarr.nonzero()[0]))

            plt.imshow(labelarr.sum(axis=1))
            plt.show()

            print('Opening Calculation, elapsed time is %d seconds'%(time.time()-time1))
            time1 = time.time()
            # label
            print(labelarr[labelarr.nonzero()])
            #exit(1)
            labelarr = skimage.measure.label(labelarr, connectivity=2)

            print('Max Connected bone labeled, elapsed time is %d seconds'%(time.time()-time1))
            time1 = time.time()



            # cluster, sort
            temp_df, cluster_df = LabelArrPixel2DataFrame(labelarr, self.array, sort=True,
                                                          keepThresholds=True, Threholds_Min=2000)
            print('cluster,sort,delete least, elapsed time is %d seconds'%(time.time()-time1))
            time1 = time.time()

            debug_temp_list = []
            Rib_Classes = []
            debug_temp_df = pd.DataFrame({})
            for index, e in cluster_df[1:].iterrows():
                #objgraph.show_growth()
                #print('hello world:',e['c'])
                singlebone = Bone(bone_data=temp_df[temp_df['c'] == e['c']], shape=labelarr.shape, naturally_shedding=True)
                singlebone.set_bone_type()
                singlebone.print_bone_info()

                #print("VOLUME:",singlebone.bonevolume)
                #plt.imshow(DataFrame2Arr(shape=self.shape,temp_df=temp_df[temp_df['c'] == e['c']]).sum(axis=1))
                #plt.show()

                if singlebone.is_rib():
                    #print("VOLUME:",singlebone.bonevolume)
                    #plt.imshow(DataFrame2Arr(shape=self.shape,temp_df=temp_df[temp_df['c'] == e['c']]).sum(axis=1))
                    #show_point = np.array(singlebone.get_joint_point())
                    #plt.scatter(x=show_point[:,2],y=show_point[:,0],c='r')
                    #plt.show()
                    #if abs(singlebone.get_joint_point()[0][2] - self.shape[2]/2) < 100:
                    debug_temp_list.extend(singlebone.get_joint_point())
                    Rib_Classes.append(e['c'])
                    #debug_temp_df = debug_temp_df.append(singlebone.get_rawdata().copy(deep=True))
                del singlebone
                gc.collect()
                objgraph.show_growth()

            plot_separater_bone(temp_df[temp_df['c'] == cluster_df['c'][0]], temp_df[temp_df['c'].isin(Rib_Classes)],self.shape, debug_temp_list)

            all_joint_list.extend(debug_temp_list)



            print('Find ribs and its joint point, elapsed time is %d seconds'%(time.time()-time1))
            #time1 = time.time()

            print('Iteration %d finished, elapsed time is %d seconds:'%(iter,time.time()-time0))

        # use biggest spine and rib, cut

        #cut bones by joint point
        print("explsion!!!")
        ribArr = self.array
        ribArr[ribArr < BoneThreshold] = 0
        ribArr[ribArr >= BoneThreshold] = 1
        #for i in all_joint_list:
            #print("explosion point:",i)
            #ribArr[i[0]-30:i[0]+30,i[1]-100:i[1]+100:,i[2]] = 0

        #store all rib
        rib_df = pd.DataFrame({})
        ribArr = skimage.measure.label(ribArr,connectivity=2)
        temp_df, cluster_df = LabelArrPixel2DataFrame(ribArr, self.array, sort=True,
                                                          keepThresholds=True, Threholds_Min=2000)

        for index, e in cluster_df[1:].iterrows():
                objgraph.show_growth()
                #print('hello world:',e['c'])
                singlebone = Bone(bone_data=temp_df[temp_df['c'] == e['c']], shape=labelarr.shape, naturally_shedding=True)
                singlebone.set_bone_type()
                singlebone.print_bone_info()
                if singlebone.is_rib():
                    rib_df = rib_df.append(temp_df[temp_df['c'] == e['c']])
                    Rib_Classes.append(e['c'])
                    #debug_temp_df = debug_temp_df.append(singlebone.get_rawdata().copy(deep=True))
                del singlebone
                gc.collect()

        plot_separater_bone(temp_df[temp_df['c'] == cluster_df['c'][0]], temp_df[temp_df['c'].isin(Rib_Classes)],self.shape, all_joint_list)
        #plt.imshow(DataFrame2Arr(shape=self.shape,temp_df=rib_df).sum(axis=1))
        #all_joint_list = np.array(all_joint_list)
        #plt.scatter(x=all_joint_list[:,2],y=all_joint_list[:,0],c='r')
        #plt.show()



"""
    def GetRibs(self):


        temp_index = labelarr.nonzero()
        tempdf = pd.DataFrame({'z':temp_index[0]})
        tempdata = self.array.copy()

    def getLargestBone(self,col_x='x',col_y='y'):
        #对当前数据分类
        self.array = sm.label(self.array, connectivity=2)
        self.df = self.array2df(self.array)
        max_cluster = self.df['v'].value_counts().index[0]
        #生成最大类
        self.LargestBone = self.df[self.df['v']==max_cluster]
        self.x_lenght_large_bone = self.LargestBone[col_x].max() - self.LargestBone[col_x].min()
        self.y_lenght_large_bone = self.LargestBone[col_y].max() - self.LargestBone[col_y].min()
        #收集散落肋骨
        self.addBoneToBoneList(self.df[self.df['v']!=max_cluster])

        self.df = self.LargestBone
        self.array = self.df2array()
        pass

    #每次对最大类做开运算
    def opening(self):
        #将当前array化成binary
        self.array[self.array>0] = 1
        self.array = sm.opening(self.array,sm.ball(1))
        pass


    #传入DataFrame(带有分类)，根据分类和条件得到rib
    def addBoneToBoneList(self,df_data, threshold_min=2000, threshold_max=350000):
        dfGroupByC = df_data.groupby('v').agg({'x': ['mean', 'min', 'max'], 'y': ['mean', 'min', 'max'],
                                                 'z': ['mean', 'min', 'max'],'v':'count'})
        dfGroupByC.columns = ['%s.%s' % e for e in self.df.columns.tolist()]
        dfGroupByC['x.length'] = dfGroupByC['x.max'] - dfGroupByC['x.min'] + 1
        dfGroupByC['y.length'] = dfGroupByC['y.max'] - dfGroupByC['y.min'] + 1
        dfGroupByC['z.length'] = dfGroupByC['z.max'] - dfGroupByC['z.min'] + 1

        #设置筛选条件得到rib
        dfGroupByC = dfGroupByC[dfGroupByC['v.count']>threshold_min & dfGroupByC['v.count']<threshold_max]
        for i in dfGroupByC['v']:
            self.rib.append(rib(df_data[df_data['v']==i]))
        pass

    #dataframe转array

    #array转dataframe
    def array2df(self,array):
        index_z, index_x, index_y = array.nonzero()
        df = pd.DataFrame({'x':index_x,'y':index_y, 'z':index_z, 'v':array[index_z,index_x,index_y]})
        return df

    def RemoveRun(self,threshold=150):
        while self.x_lenght_large_bone > threshold and self.y_lenght_large_bone > threshold:
            self.getLargestBone(self)
            self.opening(self)
"""
