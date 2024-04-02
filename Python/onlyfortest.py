import numpy as np
from tqdm import tqdm 
import torch
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append("BPS_BPTK\\Python") 
from BPS_init_function_MultiParas_torch import BPS_BPTK_MultiParas_torch

from BPS_init_function_MultiParas import BPS_BPTK_MultiParas
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
from sklearn.metrics import r2_score
import time
import scipy.stats as stats
 
## 根据定义截断分布的上下界、均值方差, 取出size个值
 
# 定义截断分布的上下界（这个只是定义的，会根据均值方差进行调整到真实的上下界），均值方差
lower, upper, mean, std = -1, 1, 0.75, 0.25
# 取到的真实的值的上下界（0.5-1）
value_lower, value_upper = mean + lower * std, mean + upper * std
print(f"True value lower and upper is: {value_lower}, {value_upper}")
# 进行截断


CV = 0.3 #变异系数
mean1 = 17.28
std1 = CV/mean1
lower1, upper1 = mean1-1.96*std1 , mean1+1.96*std1

mean2 = 6.39
std2 = CV/mean2
lower2, upper2 = mean2-1.96*std2 , mean2+1.96*std2

mean3 = 5.7
std3 = CV/mean3
lower3, upper3 = mean3-1.96*std3 , mean3+1.96*std3
X = stats.truncnorm(-2, 2, loc=mean2, scale=2.4)
# 在截断分布中取15个值
x = X.rvs(size = 15)

print(x)
