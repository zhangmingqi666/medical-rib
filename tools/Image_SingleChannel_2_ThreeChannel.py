
from PIL import Image
import numpy as np
import os


def arr_from_1c_2_3c(arr1c):
    image = np.expand_dims(arr1c, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    return image


folder_dir = '/Users/jiangyy/voc2007/JPEGSImages'

for f in os.listdir(folder_dir):

    f_path = os.path.join(folder_dir, f)
    if not os.path.exists(f_path):
        continue
    if not f_path.endswith('.jpg'):
        continue
    arr_1d = np.array(Image.open(f_path))
    print("transfer from 1c to 3c,", arr_1d.shape)
    arr_3d = arr_from_1c_2_3c(arr_1d)
    Image.fromarray(arr_3d).save(f_path)