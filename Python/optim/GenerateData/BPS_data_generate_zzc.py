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

X1 = stats.truncnorm(-1.96, 1.96, loc=mean1, scale=mean1*0.3)
x1 = X1.rvs(size = 32,random_state=42)
X2 = stats.truncnorm(-1.96, 1.96, loc=mean2, scale=mean2*0.3)
x2 = X2.rvs(size = 30,random_state=42)
X3 = stats.truncnorm(-1.96, 1.96, loc=mean3, scale=mean3*0.3)
x3 = X3.rvs(size = 30,random_state=42)

DSC_range = x1
PFO_range = x2
u1_range = x3
parameter_range = np.array(list(itertools.product(DSC_range, PFO_range, u1_range)))

BPS_plasma_data = np.zeros((32*30*30,68)) #提前分配内存
print(np.shape(parameter_range))
np.save("Python\optim\DataFromBPTK\labels_zzc.npy",parameter_range)
plasma,urinebps,urinebpsg = BPS_BPTK_MultiParas(t = time,volunteer_ID =id, paras=parameter_range ,mode = '63')

print(np.shape(urinebpsg))
print(np.shape(urinebps))
print(np.shape(plasma))

np.save("Python\\optim\\DataFromBPTK\\plasma_zzc.npy",plasma)
np.save("Python\\optim\\DataFromBPTK\\urinebps_zzc.npy",urinebps)
np.save("Python\\optim\\DataFromBPTK\\urinebpsg_zzc.npy",urinebpsg)