import numpy as np
from tqdm import tqdm 
import itertools
import torch
import numpy as np
import os
import cv2
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
# 创建一个 5x4 的二维数组


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size )

    def forward(self, x):
        out, _ = self.lstm(x)    
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out

# 神经网络参数设置
input_size = 1
hidden_size = 128
num_layers = 2
output_size = 3
dropout_prob = 0.2
learning_rate = 0.001
num_epochs = 100

X = np.load("Python\optim\BPSplasma_init_Data.npy")  #输入数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

id = 0
time = np.arange(0,75,0.005)

example_True_3para = np.array([17.28, 6.39, 5.7])

result_True = BPS_BPTK(t = time,volunteer_ID =id, DSC_0=example_True_3para[0], PFO_0=example_True_3para[1], u1_0=example_True_3para[2] ,mode = '63')
sampling_time_range = np.hstack((np.arange(0.5,20,0.5),20,np.arange(20.5,75,2))) #采样时间节点，在0至75小时内共选取了68个时间节点 
sampling_time_index = (200*sampling_time_range).astype(int) #采样时间节点在求解器结果中的索引值

X_test = result_True[sampling_time_index,25]
y_test = example_True_3para

X_test = np.reshape(X_test, (1, 68))
X_test = scaler.transform(X_test)

X_test = np.reshape(X_test, (1, 68, 1))
y_test = np.reshape(y_test, (1, 3))

X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()



test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1,shuffle = None)

best_model = CustomLSTM(input_size, hidden_size, num_layers, output_size)
best_model.load_state_dict(torch.load('Python\optim\model_best.pth'))

for test_inputs, test_labels in test_loader:  
    test_outputs = best_model(test_inputs)
    example_FromNN_3para = test_outputs

example_FromNN_3para=example_FromNN_3para.detach().numpy()
example_FromNN_3para = example_FromNN_3para.flatten()
print(example_FromNN_3para)

result_FromNN = BPS_BPTK(t = time,volunteer_ID =id, DSC_0=example_FromNN_3para[0], PFO_0=example_FromNN_3para[1], u1_0=example_FromNN_3para[2] ,mode = '63')


plt.plot(time,result_FromNN[:,25],label = 'FromNN_result')
plt.plot(time,result_True[:,25],label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()
plt.show()

plt.plot(time,abs(result_FromNN[:,25]-result_True[:,25]))
plt.xlabel('time(h)')
plt.ylabel('Absolute Error of TRUE and FROM-NN Results')
plt.show()

plt.plot(time,abs(result_FromNN[:,25]/result_True[:,25]-1),label = 'Python_result')
plt.xlabel('time(h)')
plt.ylabel('Relative Error of TRUE and FROM-NN Results')
plt.show()