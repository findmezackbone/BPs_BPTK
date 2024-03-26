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
sys.path.append("Python") 
from test_bunchofNewData_byNN_function import test_NewData_NN
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

# 定义神经网络模型
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, num_layers, output_size, dropout_prob):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_size1, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, output_size )
        #self.fc3 = nn.Linear(hidden_size3, output_size ) 
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x):
        out, _ = self.lstm(x) 
        out = self.fc1(out[:, -1, :])  
        out = self.fc2(out)
        return out

# 神经网络参数设置
input_size = 1
hidden_size1 = 128
hidden_size2 = 64
#hidden_size3 = 32
num_layers = 3
output_size = 3
learning_rate = 0.001
num_epochs = 350
dropout_prob = 0

bestmodel = CustomLSTM(input_size, hidden_size1, num_layers, output_size, dropout_prob)
best_model_path ='Python\\optim\\NNmodel_1_5\\model_1_5.pth'

Data_origin = np.load("Python\\optim\\BPSplasma_init_Data.npy")  #输入数据
time_range = np.hstack((np.arange(0.5,20,0.5),20,np.arange(20.5,75,2))) #采样时间节点，在0至75小时内共选取了68个时间节点 
paras = np.array([[17.28, 6.39, 5.7],[14.91,2.78,4.707],[21.333, 9.666, 7.42],[19.921, 11, 2.4212]])
time = np.arange(0,75,0.005)

mean,result_FromNN_Total,result_True_Total = test_NewData_NN(origin_para = paras , model = bestmodel, model_path = best_model_path, Data = Data_origin, sampling_time_range = time_range)
print(mean)

plt.figure(figsize=(8,8), dpi=80)
plt.figure(1)

plt.subplot(2,2,1)
plt.plot(time,result_FromNN_Total,label = 'FromNN_result')
plt.plot(time,result_True_Total,label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()

plt.subplot(2,2,2)
plt.plot(time,abs(result_FromNN_Total-result_True_Total))
plt.xlabel('time(h)')
plt.ylabel('Absolute Error of TRUE and FROM-NN Results')

plt.subplot(2,2,3)
plt.plot(time[1:],abs(result_FromNN_Total[1:]/result_True_Total[1:]-1))
plt.xlabel('time(h)')
plt.ylabel('Relative Error of TRUE and FROM-NN Results')


plt.show()