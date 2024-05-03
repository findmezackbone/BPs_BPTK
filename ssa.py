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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(torch.__version__)
print(torch.backends.cudnn.enabled)

print(torch.version.cuda)
print(device)