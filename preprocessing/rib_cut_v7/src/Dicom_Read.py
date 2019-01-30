#coding=utf-8
import pydicom as dicom
import os
import numpy as np
import scipy
import scipy.ndimage
import gc
import cv2 as cv
import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
# Load the scans in given folder path
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
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


def resample(image, scan, new_spacing=[1, 1, 1]):
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
    if real_resize_factor[0] == 1:
        new_image = np.zeros(tuple([int(i) for i in new_shape]))
        for i in range(int(new_shape[0])):
            new_image[i] = cv.resize(image[i], (0, 0), fx=real_resize_factor[1], fy=real_resize_factor[2], interpolation=cv.INTER_CUBIC)
    else:
        new_image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    del image
    gc.collect()
    return new_image, new_spacing


def read_dcm_info():
    pass


class RibDataFrame:
    def __init__(self):
        self.Data = None
        #pass

    def readDicom(self, path=None):

        with timer("read_folder_name"):
            folder_name = os.path.abspath(path)

        with timer("first_patient"):
            first_patient = load_scan(folder_name)

        with timer("first_patient_pixels"):
            first_patient_pixels = get_pixels_hu(first_patient)
        gc.collect()

        with timer("pix_resampled"):
            pix_resampled, _ = resample(first_patient_pixels, first_patient, [1, 1, 1])

        return pix_resampled


if __name__=='__main__':
    pass
    """
    pix_resampled = RibDataFrame().readDicom(path='/Users/jiangyy/projects/medical-rib/dataset_first/A1076956')
    print(pix_resampled.shape)
    pix_resampled[pix_resampled < 400] = 0
    pix_resampled[pix_resampled >= 400] = 1
    plt.imshow(pix_resampled.sum(axis=2))
    plt.show()
    """


