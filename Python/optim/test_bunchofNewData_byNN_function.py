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
from BPS_init_function import BPS_BPTK 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
from sklearn.metrics import r2_score

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
best_model_path = 'Python\\optim\\model_pause1.pth'

Data_origin = np.load("Python\optim\BPSplasma_init_Data.npy")  #输入数据
time_range = np.hstack((np.arange(0.5,20,0.5),20,np.arange(20.5,75,2))) #采样时间节点，在0至75小时内共选取了68个时间节点 
paras = np.array([[17.28, 6.39, 5.7]])


def test_NewData_NN(origin_para = paras , model = bestmodel, model_path = best_model_path, Data = Data_origin, sampling_time_range = time_range):

    scaler = StandardScaler()
    X = scaler.fit_transform(Data)

    best_model = model
    best_model.load_state_dict(torch.load(model_path))

    id = 0
    time = np.arange(0,75,0.005)
    sampling_time_index = (200*sampling_time_range).astype(int) #采样时间节点在求解器结果中的索引值

    result_True_Total = np.zeros((1,15000)).flatten()
    result_FromNN_Total = np.zeros((1,15000)).flatten()
    result_True_Total_Adjusted = np.zeros((1,15000)).flatten()
    result_FromNN_Total_Adjusted = np.zeros((1,15000)).flatten()

    norm1_error = np.zeros((np.shape(origin_para)[0],2))
    r2 = np.zeros((np.shape(origin_para)[0],1)).flatten()

    for i in range(np.shape(origin_para)[0]):
        result_True = BPS_BPTK(t = time,volunteer_ID =id, DSC_0=origin_para[i,0], PFO_0=origin_para[i,1], u1_0=origin_para[i,2] ,mode = '63')
        result_True_Total = result_True[:,25]+ result_True_Total
        result_True_Total_Adjusted = result_True[:,25]/np.mean(result_True[:,25])*15000 + result_True_Total_Adjusted
        

        X_test = result_True[sampling_time_index,25]
        y_test = origin_para[i,:]

        X_test = np.reshape(X_test, (1, len(sampling_time_range)))
        X_test = scaler.transform(X_test)

        X_test = np.reshape(X_test, (1, len(sampling_time_range), 1))
        y_test = np.reshape(y_test, (1, 3))

        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()



        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=1,shuffle = None)

        
    
        for test_inputs, test_labels in test_loader:  
            test_outputs = best_model(test_inputs)
            example_FromNN_3para = test_outputs

        example_FromNN_3para=example_FromNN_3para.detach().numpy()
        example_FromNN_3para = example_FromNN_3para.flatten()

        result_FromNN = BPS_BPTK(t = time,volunteer_ID =id, DSC_0=example_FromNN_3para[0], PFO_0=example_FromNN_3para[1], u1_0=example_FromNN_3para[2] ,mode = '63')
        result_FromNN_Total_Adjusted = result_FromNN[:,25]/np.mean(result_FromNN[:,25])*15000 + result_FromNN_Total_Adjusted
        result_FromNN_Total = result_FromNN[:,25] + result_FromNN_Total

        r2[i] = r2_score(result_True[:,25], result_FromNN[:,25]) #决定系数

        norm_absolute = np.linalg.norm(result_FromNN[:,25]-result_True[:,25], ord=1)/15000
        norm_relative = np.linalg.norm(result_FromNN[1:,25]/result_True[1:,25]-1, ord=1)/14999

        norm1_error[i,0] = norm_absolute
        norm1_error[i,1] = norm_relative
        
    mean_r2 = np.mean(r2)#平均决定系数

    mean_abs_err =  np.mean(norm1_error[:,0])
    mean_rel_err =  np.mean(norm1_error[:,1])

    mean_err= np.hstack((mean_abs_err,mean_rel_err))

    return mean_err,mean_r2,result_FromNN_Total/np.shape(origin_para)[0],result_True_Total/np.shape(origin_para)[0],result_FromNN_Total_Adjusted/np.shape(origin_para)[0],result_True_Total_Adjusted/np.shape(origin_para)[0]


