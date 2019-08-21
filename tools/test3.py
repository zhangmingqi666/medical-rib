
from scipy.io import loadmat
m = loadmat("/Users/jiangyy/Downloads/SUNRGBDMeta3DBB_v2.mat")
print(m['SUNRGBDMeta'][0, 0])