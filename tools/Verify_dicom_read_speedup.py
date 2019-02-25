
"""
@2.21 make better the resample algorithm in the dicom read.
So here We must Verify it works.
show2d
"""
import os
import pickle as pkl
import matplotlib.pyplot as plt
pkl_path = "/Users/jiangyy/DataSet/rib_dataSet/updated48labeled_1.31/pkl_cache"
files = os.listdir(pkl_path)

for file in files:
    f_path = os.path.join(pkl_path, file)
    rib_data = pkl.load(open(f_path, 'rb'))
    rib_data[rib_data <= 400] = 0
    rib_data[rib_data > 400] = 1
    plt.imshow(rib_data.sum(axis=1))
    save_f_name = file.replace('.pkl', '.png')
    plt.savefig(os.path.join('/Users/jiangyy/projects/medical-rib/tools/Verify_logs/Verify_dicom_read_speedup', save_f_name))



