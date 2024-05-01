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


X0 = np.load("Python\optim\DataFromBPTK\plasma28\plasma28_SG.npy")  #输入数据
X1 = np.load("Python\optim\DataFromBPTK\\urine15\\urinebps15_SG.npy")
X2 = np.load("Python\optim\DataFromBPTK\\urine15\\urinebpsg15_SG.npy")
X =  np.hstack((X0,X1,X2))
print(np.shape(X))
np.save("Python\optim\DataFromBPTK\plasma28+urine15x2\plasma28+urine15x2_SG.npy",X)