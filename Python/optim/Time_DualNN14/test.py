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

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

X =  np.load("Python\optim\Time_DualNN14\database\input_add.npy")  
y = np.load("Python\optim\Time_DualNN14\database\label_add.npy")  

y = torch.tensor(y)
print(X.shape)
print(y.shape[0])