import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("Python") 
from BPS_init_function_MultiParas import BPS_BPTK_MultiParas
import itertools
from tqdm import tqdm 
import torch
import scipy.stats as stats

#这个文件是用于生成1号志愿者用模型计算出的指定时间节点的血药浓度的数据集(代入10000组不同的待反演参数，得到10000条数据)

id = 0
time = np.arange(0,75,0.005) #七十五个小时的时间戳
CV = 0.3 #变异系数
mean1 = 17.28
mean2 = 6.39
mean3 = 5.7

X1 = stats.truncnorm(-2, 2, loc=mean1, scale=2.4)
x1 = X1.rvs(size = 30)
X2 = stats.truncnorm(-2, 2, loc=mean2, scale=2.4)
x2 = X2.rvs(size = 25)
X3 = stats.truncnorm(-2, 2, loc=mean3, scale=2.4)
x3 = X3.rvs(size = 25)

DSC_range = x1
PFO_range = x2
u1_range = x3
parameter_range = np.array(list(itertools.product(DSC_range, PFO_range, u1_range)))
sampling_time_range = np.hstack((np.arange(0.5,20,0.5),20,np.arange(20.5,75,2))) #采样时间节点，在0至75小时内共选取了68个时间节点 
sampling_time_index = (200*sampling_time_range).astype(int) #采样时间节点在求解器结果中的索引值

BPS_plasma_data = np.zeros((30*25*25,68)) #提前分配内存


result = BPS_BPTK_MultiParas(t = time,volunteer_ID =id, paras=parameter_range ,mode = '63')
BPS_plasma_data = result[:,sampling_time_index]

np.save("Python\optim\DataFromBPTK\BPSplasma_init_Data_addition.npy",BPS_plasma_data)