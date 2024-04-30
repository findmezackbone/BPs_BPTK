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
sys.path.append("Python") 
from BPS_init_function import BPS_BPTK 
import matplotlib.pyplot as plt
import keyboard


X0 = np.load("Python\optim\DataFromBPTK\\urinebpsg5_1_zzc.npy")  #输入数据
X1 = np.load("Python\optim\DataFromBPTK\\urinebpsg5_1_SG.npy")  #输入数据
#X2 = np.load("Python\optim\DataFromBPTK\\urine15\\urinebps15_zzc.npy")
#X3 = np.load("Python\optim\DataFromBPTK\\urine15\\urinebpsg15_zzc.npy")
X =  np.vstack((X0,X1))
print(np.shape(X))
np.save("Python\optim\DataFromBPTK\\urinebpsg5_1_add.npy",X)