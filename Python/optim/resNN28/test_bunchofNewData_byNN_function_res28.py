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
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append("Python") 
from BPS_init_function import BPS_BPTK 
from BPS_init_function_MultiParas import BPS_BPTK_MultiParas
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
from sklearn.metrics import r2_score

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
hyperparas = {'input_dim':28,'hidden_dim':30,'hidden_nums':3,'output_dim':3,'block_layer_nums':3}
learning_rate = 0.001
num_epochs = 100

bestmodel = CustomResNN(hyperparas)
best_model_path = 'Python\\optim\\Temporary_Model\\model_best.pth'

Data_origin = np.load("Python\optim\DataFromBPTK\plasma28\plasma28_zzc.npy")  #输入数据
a = np.arange(0.5,5.1,0.5)
b = np.arange(6,15.1,1)
c = np.arange(18,42.1,6)
d = np.array([50,60,72])
time_range = np.hstack((a,b,c,d)) #采样时间节点，在0至75小时内共选取了28个时间节点 

paras = np.array([[17.28, 6.39, 5.7]])

criterion = nn.MSELoss()

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

    result_True,_,_ =BPS_BPTK_MultiParas(t = time,volunteer_ID =id, paras = origin_para ,mode = '63')
    result_True_Total = np.sum(result_True, axis=0)
    for i in range(np.shape(origin_para)[0]):
        result_True_Total_Adjusted = result_True[i,:]/np.mean(result_True[i,:])*15000 + result_True_Total_Adjusted

    X_test = result_True[:,sampling_time_index]
    y_test = origin_para

    X_test = np.reshape(X_test, (np.shape(origin_para)[0], len(sampling_time_range)))
    X_test = scaler.transform(X_test)

    
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()



    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=np.shape(origin_para)[0],shuffle = None)    


    for test_inputs, test_labels in test_loader:  
        test_outputs = best_model(test_inputs)
        example_FromNN_3para = test_outputs
        loss = criterion(test_outputs, test_labels)/len(test_inputs)

        
        label_rel_err = test_outputs/test_labels-1
        label_rel_err_mean = torch.sum(label_rel_err,dim=0)/np.shape(origin_para)[0]

    example_FromNN_3para=example_FromNN_3para.detach().numpy()
    
    result_FromNN,_,_ = BPS_BPTK_MultiParas(t = time,volunteer_ID =id, paras = example_FromNN_3para , mode = '63')

    err28 = np.zeros((np.shape(origin_para)[0],2))
    result_FromNN_Total = np.sum(result_FromNN, axis=0)
    for i in range(np.shape(origin_para)[0]):
        result_FromNN_Total_Adjusted = result_FromNN[i,:]/np.mean(result_True[i,:])*15000 + result_FromNN_Total_Adjusted
        r2[i] = r2_score(result_True[i,:], result_FromNN[i,:]) #决定系数
        norm_absolute = np.linalg.norm(result_FromNN[i,:]-result_True[i,:], ord=1)/15000
        norm_relative = np.linalg.norm(result_FromNN[i,1:]/(result_True[i,1:]+1E-16)-1, ord=1)/14999
        #print(norm_relative)
        norm1_error[i,0] = norm_absolute
        norm1_error[i,1] = norm_relative
        
        err28[i,0] =  np.linalg.norm(result_FromNN[i,sampling_time_index]-result_True[i,sampling_time_index], ord=1)/np.shape(sampling_time_index)[0]
        err28[i,1] =  np.linalg.norm(result_FromNN[i,sampling_time_index]/(result_True[i,sampling_time_index]+1E-16)-1, ord=1)/np.shape(sampling_time_index)[0]

    mean_r2 = np.mean(r2)#平均决定系数
    mean_abs_err =  np.mean(norm1_error[:,0])
    mean_rel_err =  np.mean(norm1_error[:,1])
    mean_err28 = np.mean(err28,axis=0)
    mean_err= np.hstack((mean_abs_err,mean_rel_err))

    return mean_err28 ,loss,label_rel_err_mean,mean_err,mean_r2,result_FromNN_Total/np.shape(origin_para)[0],result_True_Total/np.shape(origin_para)[0],result_FromNN_Total_Adjusted/np.shape(origin_para)[0],result_True_Total_Adjusted/np.shape(origin_para)[0]


