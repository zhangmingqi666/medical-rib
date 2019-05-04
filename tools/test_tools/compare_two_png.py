import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
import numpy as np
import xml.etree.ElementTree as ET
import os


png_old_folder = "/Users/jiangyy/projects/medical-rib/tools/results_for_problems/unavail.last"
png_new_folder = "/Users/jiangyy/projects/medical-rib/tools/results_for_problems/unavail"
diff_folder = "/Users/jiangyy/projects/medical-rib/tools/results_for_problems/unavail_few"

for file in os.listdir(png_old_folder):
    print(file)
    old_image_path = "{}/{}".format(png_old_folder, file)
    new_image_path = "{}/{}".format(png_new_folder, file)
    diff_image_path = "{}/{}".format(diff_folder, file)

    plt.figure(figsize=(12, 4))
    grid = plt.GridSpec(4, 12, wspace=0.5, hspace=0.5)

    plt.subplot(grid[0:4, 0:4])
    image = Image.open(old_image_path)
    plt.imshow(image)

    plt.subplot(grid[0:4, 4:8])
    image = Image.open(new_image_path)
    plt.imshow(image)

    if os.path.exists(diff_image_path):
        plt.subplot(grid[0:4, 8:12])
        image = Image.open(diff_image_path)
        plt.imshow(image)
    plt.show()
