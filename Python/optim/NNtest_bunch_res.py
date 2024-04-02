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
from test_bunchofNewData_byNN_function_res import test_NewData_NN
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
from sklearn.metrics import r2_score
import scipy.stats as stats

# 定义神经网络模型
class ResNetBlock(nn.Module):
    def __init__(self, hyperparas):
        super(ResNetBlock, self).__init__()
        
        self.hidden_dim = hyperparas['hidden_dim']
        self.block_layer_nums =hyperparas['block_layer_nums']
            
        # Define layers for the function f (MLP)
        self.layers = nn.ModuleList()
        
        for _ in range(self.block_layer_nums - 1):  # -2 because we already added one layer and last layer is already defined
            self.layers.append(nn.Linear(self.hidden_dim,self.hidden_dim ))
        
        # Layer normalization
        self.layernorms = nn.ModuleList()
        for _ in range(self.block_layer_nums - 1):  # -1 because layer normalization is not applied to the last layer
            self.layernorms.append(nn.LayerNorm(self.hidden_dim))
        
    def forward(self, x):
        # Forward pass through the function f (MLP)
        out = x
        for i in range(self.block_layer_nums - 1):  # -1 because last layer is already applied outside the loop
            out = self.layers[i](out)
            out = self.layernorms[i](out)
            out = torch.relu(out)
        
        # Element-wise addition of input x and output of function f(x)
        out = x + out
        
        return out
    

class CustomResNN(nn.Module):
    def __init__(self,hyperparas):
        super().__init__()
        self.input_dim = hyperparas['input_dim'] #68
        self.hidden_dim = hyperparas['hidden_dim'] #30
        self.hidden_nums = hyperparas['hidden_nums'] #3
        self.output_dim = hyperparas['output_dim'] #3
        self.block_layer_nums = hyperparas['block_layer_nums'] #3

        self.layer_list = []
        self.layer_list.append(nn.Sequential(nn.Linear(self.input_dim,self.hidden_dim),nn.ReLU() ) )

        for _ in range(self.hidden_nums-1):
            self.layer_list.append(ResNetBlock(hyperparas))

        self.layer_list.append(nn.Linear(self.hidden_dim,self.output_dim))

        self.linear_Res_final = nn.Sequential(*self.layer_list)

    def forward(self,inputs):
        
        return self.linear_Res_final(inputs)

#超参数合集
hyperparas = {'input_dim':68,'hidden_dim':30,'hidden_nums':3,'output_dim':3,'block_layer_nums':3}
learning_rate = 0.001
num_epochs = 100

bestmodel = CustomResNN(hyperparas)

best_model_path ='Python\optim\Temporary_Model\model_pause1.pth'
best_model_path ='Python\optim\Temporary_Model\model_pause2.pth'
best_model_path ='Python\optim\Temporary_Model\model_pause3.pth'
#best_model_path ='Python\optim\Temporary_Model\model_best.pth'

Data_origin = np.load("Python\\optim\\DataFromBPTK\\BPSplasma_init_Data_2.0.npy")  #输入数据
time_range = np.hstack((np.arange(0.5,20,0.5),20,np.arange(20.5,75,2))) #采样时间节点，在0至75小时内共选取了68个时间节点

mean1 = 17.28
mean2 = 6.39
mean3 = 5.7

X1 = stats.truncnorm(-1.8, 2.3, loc=mean1, scale=5)
x1 = X1.rvs(size = 6,random_state = 43)
X2 = stats.truncnorm(-1, 1.3, loc=mean2, scale=5)
x2 = X2.rvs(size = 5,random_state = 43)
X3 = stats.truncnorm(-1, 1.2, loc=mean3, scale=5)
x3 = X3.rvs(size = 5,random_state = 43)

paras = np.array(list(itertools.product(x1, x2, x3)))
#paras = np.array([[17.28, 6.39, 5.7],[14.91,2.78,4.707],[21.333, 9.666, 7.42],[19.921, 11, 2.4212],[23.032,7.2223,4.99],[13.53,10.11,5.444],[18.888,9.999,8.888],[15.5,10.5,7.5],[12.45,6.84,6.75],[20.8,4.61,9.888]])
time = np.arange(0,75,0.005)

mean_err,mean_r2,result_FromNN_Total,result_True_Total,result_FromNN_Total_Ajusted,result_True_Total_Ajusted = test_NewData_NN(origin_para = paras , model = bestmodel, model_path = best_model_path, Data = Data_origin, sampling_time_range = time_range)
print(mean_err)
print(mean_r2)
#print(np.mean(result_True_Total))
print(mean_err[0]/np.mean(result_True_Total))

plt.plot(time,result_FromNN_Total,label = 'FromNN_result')
plt.plot(time,result_True_Total,label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()
plt.show()

plt.plot(time,abs(result_FromNN_Total-result_True_Total))
norm_absolute = np.linalg.norm(result_FromNN_Total-result_True_Total, ord=1)/15000
norm_absolute = '%.4g' % norm_absolute
print(norm_absolute )
plt.annotate(f'norm1 of abs_err:{norm_absolute}',xy = [0.3,0.8], xycoords='figure fraction',weight='bold',color = 'blue')
plt.xlabel('time(h)')
plt.ylabel('Absolute Error of TRUE and FROM-NN Results')
plt.show()



plt.plot(time[1:],abs(result_FromNN_Total[1:]/result_True_Total[1:]-1))
norm_relative = np.linalg.norm(result_FromNN_Total[1:]/result_True_Total[1:]-1, ord=1)/15000
norm_relative = '%.4g' % norm_relative
print(norm_relative)
plt.annotate(f'norm1 of rel_err:{norm_relative}',xy = [0.3,0.8], xycoords='figure fraction',weight='bold',color = 'blue')
plt.xlabel('time(h)')
plt.ylabel('Relative Error of TRUE and FROM-NN Results')


plt.show()

plt.plot(time,result_FromNN_Total_Ajusted,label = 'FromNN_result')
plt.plot(time,result_True_Total_Ajusted,label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()
plt.show()

plt.plot(time,abs(result_FromNN_Total_Ajusted-result_True_Total_Ajusted))
norm_absolute = np.linalg.norm(result_FromNN_Total_Ajusted-result_True_Total_Ajusted, ord=1)/15000
norm_absolute = '%.4g' % norm_absolute
print(norm_absolute )
plt.annotate(f'norm1 of abs_err:{norm_absolute}',xy = [0.3,0.8], xycoords='figure fraction',weight='bold',color = 'blue')
plt.xlabel('time(h)')
plt.ylabel('Absolute Error of TRUE and FROM-NN Results')
plt.show()



plt.plot(time[1:],abs(result_FromNN_Total_Ajusted[1:]/result_True_Total_Ajusted[1:]-1))
norm_relative = np.linalg.norm(result_FromNN_Total_Ajusted[1:]/result_True_Total_Ajusted[1:]-1, ord=1)/15000
norm_relative = '%.4g' % norm_relative
print(norm_relative)
plt.annotate(f'norm1 of rel_err:{norm_relative}',xy = [0.3,0.8], xycoords='figure fraction',weight='bold',color = 'blue')
plt.xlabel('time(h)')
plt.ylabel('Relative Error of TRUE and FROM-NN Results')


plt.show()