import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("Python") 
from BPS_init_function_MultiParas import BPS_BPTK_MultiParas
from SparseGrids import sparse_grids_scale
import itertools
from tqdm import tqdm 

#这个文件是用于生成1号志愿者用模型计算出的指定时间节点的血药浓度的数据集(使用sparseGrids方法来在参数空间中取样)

id = 0
time = np.arange(0,75,0.005) #七十五个小时的时间戳

grids_scale = sparse_grids_scale(dim=3 , level=9)
parameter_range = np.zeros((18943,3))

parameter_range[:,0] = 14*grids_scale[:,0]+10.28
parameter_range[:,1] = 10*grids_scale[:,1]+1.39
parameter_range[:,2] = 9*grids_scale[:,2]+1.2

sampling_time_range = np.hstack((np.arange(0.5,20,0.5),20,np.arange(20.5,75,2))) #采样时间节点，在0至75小时内共选取了68个时间节点 
sampling_time_index = (200*sampling_time_range).astype(int) #采样时间节点在求解器结果中的索引值


BPS_plasma_data = np.zeros((18943,68)) #提前分配内存
print(np.shape(parameter_range))
np.save("Python\optim\DataFromBPTK\labels_SG.npy",parameter_range)
result = BPS_BPTK_MultiParas(t = time,volunteer_ID =id, paras=parameter_range ,mode = '63')
BPS_plasma_data = result[:,sampling_time_index]

np.save("Python\optim\DataFromBPTK\BPSplasma_init_Data_addition_SG.npy",BPS_plasma_data)