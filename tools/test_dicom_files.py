import os
import pydicom as dicom

"""
There are some difference between linux and mac, mac can take the initiative to shield the file started with .
"""

path = "../dataSet/updated48labeled_1.31/dataset/41121210/STU12539050/SER1063"
slicings = []
for file in os.listdir(path):
    print(file)