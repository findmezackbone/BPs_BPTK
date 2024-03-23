import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("Python") 
from BPS_init_function import BPS_BPTK 
import itertools
from tqdm import tqdm 

#这个文件是用于生成1号志愿者用模型计算出的指定时间节点的血药浓度的数据集(代入10000组不同的待反演参数，得到10000条数据)

id = 0
time = np.arange(0,75,0.005) #七十五个小时的时间戳
DSC_range = np.hstack((np.arange(15,10,-1.2),np.arange(15.5,20,0.5),np.arange(21,30,1.2),31,34,37))
PFO_range = np.hstack((1.2,np.arange(2,8,0.4),np.arange(8,15,1.8)))
u1_range = np.hstack((1.2,2.4,np.arange(3,8,0.4),np.arange(8.8,15,4/3)))
parameter_range = np.array(list(itertools.product(DSC_range, PFO_range, u1_range)))
sampling_time_range = np.hstack((np.arange(0.5,20,0.5),20,np.arange(20.5,75,2))) #采样时间节点，在0至75小时内共选取了68个时间节点 
sampling_time_index = (200*sampling_time_range).astype(int) #采样时间节点在求解器结果中的索引值

BPS_plasma_data = np.zeros((10000,68)) #提前分配内存

for i in tqdm(range(10000)):
    result = BPS_BPTK(t = time,volunteer_ID =id, DSC_0=parameter_range[i,0], PFO_0=parameter_range[i,1], u1_0=parameter_range[i,2] ,mode = '63')
    BPS_plasma_data[i,:] = result[sampling_time_index,25]

np.save("Python\optim\BPSplasma_init_Data.npy",BPS_plasma_data)