import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("Python") 
from BPS_init_function_MultiParas import BPS_BPTK_MultiParas
from SparseGrids import sparse_grids_scale2
import itertools
from tqdm import tqdm 

#这个文件是用于生成1号志愿者用模型计算出的指定时间节点的血药浓度的数据集(使用sparseGrids方法来在参数空间中取样)

id = 0
time = np.arange(0,75,0.005) #七十五个小时的时间戳

grids_scale1 = sparse_grids_scale2(dim=2 , level=2)
grids_scale2 = sparse_grids_scale2(dim=2 , level=3)
grids_scale3 = sparse_grids_scale2(dim=2 , level=5)


plt.scatter(grids_scale1[:,0],grids_scale1[:,1])
plt.axis('equal') 
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

plt.scatter(grids_scale2[:,0],grids_scale2[:,1])
plt.axis('equal') 
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

plt.scatter(grids_scale3[:,0],grids_scale3[:,1])
plt.axis('equal') 
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()