import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema
from scipy import stats
import random
import itertools
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
sys.path.append("Python") 
import matplotlib.pyplot as plt
import keyboard
sys.path.append("Python") 
from BPS_init_function_MultiParas_copy import BPS_BPTK_MultiParas

id = 0
timel = np.arange(0,75,0.005)
a = np.array([[14.9377, 6.1184, 3.7767]])
data =BPS_BPTK_MultiParas(t = timel,volunteer_ID =id, paras =a,mode = '63')

plt.plot(timel, data[0,:,12],label='1')
plt.plot(timel, data[0,:,25],label='2')
plt.legend()
plt.show()