import os
import pydicom as dicom

path = "../dataSet/updated48labeled_1.31/dataset/41121210/STU12539050/SER1063"
slicings = []
for file in os.listdir(path):
    print(file)