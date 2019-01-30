# ***************************************************************************
# *
# * Description: label propagation
# * Author: Zou Xiaoyi (zouxy09@qq.com)
# * Date:   2015-10-15
# * HomePage: http://blog.csdn.net/zouxy09
# *
# **************************************************************************

import time
import math
import numpy as np
from label_propagation import labelPropagation
import scipy
import glob
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pydicom as dicom
import scipy.misc
import numpy as np

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]]))
    spacing = np.array(list(spacing))
    print('original spacing:{}'.format(spacing))
    resize_factor = new_spacing / spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    print('new shape:{}, real_size_facor:{}'.format(new_shape, real_resize_factor))
    new_spacing = spacing * real_resize_factor
    print('new_spacing:{}'.format(new_spacing))
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    print('jelllo')
    return image, new_spacing

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


# show
def show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = np.random.rand(Mat_Label.shape[0])
    for i in range(Mat_Unlabel.shape[0]):
        ax.scatter(Mat_Unlabel[i,0], Mat_Unlabel[i,1], Mat_Unlabel[i,1], colors=colors[unlabel_data_labels[i]])
    plt.show()

    #import matplotlib.pyplot as plt

    """
    for i in range(Mat_Label.shape[0]):
        if int(labels[i]) == 0:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dr')
        elif int(labels[i]) == 1:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Db')
        else:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dy')

    for i in range(Mat_Unlabel.shape[0]):
        if int(unlabel_data_labels[i]) == 0:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'or')
        elif int(unlabel_data_labels[i]) == 1:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'ob')
        else:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'oy')

    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.xlim(0.0, 12.)
    plt.ylim(0.0, 12.)
    plt.show()
    """


def loadCircleData(num_data):
    center = np.array([5.0, 5.0])
    radiu_inner = 2
    radiu_outer = 4
    num_inner = num_data / 3
    num_outer = num_data - num_inner

    data = []
    theta = 0.0
    for i in range(int(num_inner)):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_inner * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_inner * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        theta += 2

    theta = 0.0
    for i in range(int(num_outer)):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_outer * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_outer * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        theta += 1

    Mat_Label = np.zeros((2, 2), np.float32)
    Mat_Label[0] = center + np.array([-radiu_inner + 0.5, 0])
    Mat_Label[1] = center + np.array([-radiu_outer + 0.5, 0])
    labels = [0, 1]
    Mat_Unlabel = np.vstack(data)
    return Mat_Label, labels, Mat_Unlabel

"""
def loadBandData(num_unlabel_samples):
    # Mat_Label = np.array([[5.0, 2.], [5.0, 8.0]])
    # labels = [0, 1]
    # Mat_Unlabel = np.array([[5.1, 2.], [5.0, 8.1]])

    Mat_Label = np.array([[5.0, 2.], [5.0, 8.0]])
    labels = [0, 1]
    num_dim = Mat_Label.shape[1]
    Mat_Unlabel = np.zeros((num_unlabel_samples, num_dim), np.float32)
    Mat_Unlabel[:num_unlabel_samples / 2, :] = (np.random.rand(num_unlabel_samples / 2, num_dim) - 0.5) * np.array(
        [3, 1]) + Mat_Label[0]
    Mat_Unlabel[num_unlabel_samples / 2: num_unlabel_samples, :] = (np.random.rand(num_unlabel_samples / 2,
                                                                                   num_dim) - 0.5) * np.array([3, 1]) + \
                                                                   Mat_Label[1]
    return Mat_Label, labels, Mat_Unlabel
"""

# main function
if __name__ == "__main__":
    import numpy as np
    import pickle

    m = pickle.load(open("tmp2.txt", "rb"))
    m[m < 400] = 0
    aaa = m.nonzero()
    # print(len(aaa[0]))
    data = np.concatenate((aaa[0].reshape(-1, 1), aaa[1].reshape(-1, 1)), axis=1)
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
    # import matplotlib.pyplot as plt
    # plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow')
    # plt.show()
    # list = []
    # labelunique = np.unique(cluster.labels_)
    # for e in labelunique
    import pandas as pd

    df = pd.DataFrame({'x': data[:, 0], 'z': data[:, 1], 'label': cluster.labels_})
    average = df.groupby('label').agg({'x': 'mean', 'z': 'mean'})
    average.reset_index(inplace=True)
    average.columns = ['label', 'x_mean', 'z_mean']
    average['x_mean'] = average['x_mean'].apply(np.int)
    average['z_mean'] = average['z_mean'].apply(np.int)
    average['y_mean'] = 400
    Mat_Label = average[['x_mean','y_mean','z_mean']].values
    labels = average['label'].values
    bonenonzero = pickle.load(open("tmp.txt", "rb"))
    #bone[bone<400] = 0
    #bonenonzero = bone.nonzero()
    Mat_Unlabel = np.concatenate((bonenonzero[0].reshape(-1,1),bonenonzero[1].reshape(-1,1),bonenonzero[2].reshape(-1,1)),axis=1)
    #print(average)
    #label = np.zeros()
    #print(pix_resampled)

    #num_unlabel_samples = 800
    # Mat_Label, labels, Mat_Unlabel = loadBandData(num_unlabel_samples)
    #Mat_Label, labels, Mat_Unlabel = loadCircleData(num_unlabel_samples)
    #print(Mat_Label)
    #print(labels)
    #print(Mat_Unlabel)

    ## Notice: when use 'rbf' as our kernel, the choice of hyper parameter 'sigma' is very import! It should be
    ## chose according to your dataset, specific the distance of two data points. I think it should ensure that
    ## each point has about 10 knn or w_i,j is large enough. It also influence the speed of converge. So, may be
    ## 'knn' kernel is better!
    # unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf', rbf_sigma = 0.2)
    unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type='knn', knn_num_neighbors=10,
                                           max_iter=1)
    show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels)
