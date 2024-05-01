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
from BPS_init_function_MultiParas import BPS_BPTK_MultiParas
import matplotlib.pyplot as plt
import keyboard

def label_transform_reverse_tensor(x): #翻转预处理
    x = (torch.exp(x)-1 ) /10000
    return(x)

def standard_transform(x):
    # 计算每个特征的均值
    mean = x.mean(dim=0)
    # 计算每个特征的标准差
    std = x.std(dim=0, unbiased=False)  # unbiased=False 相当于 numpy 的 ddof=0
    # 避免使用0的标准差
    std[std == 0] = 1
    # 进行标准化转换
    x = (x - mean) / std
    return x,mean,std

def inverse_transform(x, mean, std):
    # 将标准化后的张量乘以标准差，然后加上均值
    x = x * std + mean
    return x

# 准备数据
X1 = np.load("Python\optim\DataFromBPTK\plasma28+urine15x2\plasma28+urine15x2_zzc.npy")  #输入数据
X2 = np.load("Python\optim\DataFromBPTK\plasma28+urine15x2\plasma28+urine15x2_SG.npy")  #输入数据

y1 = np.load("Python\optim\DataFromBPTK\labels_zzc.npy")
y2 = np.load("Python\optim\DataFromBPTK\labels_SG.npy")

X =  np.vstack((X1,X2))
y =  np.vstack((y1,y2))

# 数据预处理 标准化数据
X = torch.tensor(X).float()
y = torch.tensor(y).float()

X,X_mean,X_std = standard_transform(X)

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.15, random_state=20)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

# 将数据转移到 GPU（如果可用

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
class ResNN_Forward(nn.Module):
    def __init__(self,hyperparas):
        super().__init__()
        self.input_dim = hyperparas['input_dim'] #28
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
    
class ResNN_Reverse(nn.Module):
    def __init__(self,hyperparas):
        super().__init__()
        self.input_dim = hyperparas['input_dim'] #28
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
hyperparas_reverse = {'input_dim':58,'hidden_dim':30,'hidden_nums':3,'output_dim':3,'block_layer_nums':3}
hyperparas_forward = {'input_dim':3,'hidden_dim':30,'hidden_nums':3,'output_dim':28,'block_layer_nums':3}
learning_rate = 0.001
num_epochs = 300


# 初始化模型、损失函数和优化器
model = ResNN_Reverse(hyperparas_reverse).to(device)
model_forward = ResNN_Forward(hyperparas_forward).to(device)
model_forward.load_state_dict(torch.load('Python\ForwardFitNN\Settled_Model\\threeTo28\\4\model2.pth'))

loss_function=nn.MSELoss()

def criterion(output,label,input,mode = 1):
    if mode == 1:
        return loss_function(output,label)
    if mode == 2:
        output_forward = model_forward(output) #模型输出的三参数代入至PBPK的拟合网络中得到一个代表28个血液药含量采样点的数组
        output_forward = label_transform_reverse_tensor(output_forward)
        input_inverse = inverse_transform(input, X_mean, X_std) #将输入特征解除标准化
        return loss_function(output_forward,input_inverse[:,:28])
    
    


# 加载效果最好的模型
best_model = ResNN_Reverse(hyperparas_reverse).to(device)
best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_best.pth'))
best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_pause3.pth'))

# 在测试集上评估模型
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1,shuffle = None)

test_loss = 0.0
label_abs_err_total = 0.0
label_rel_err_total = 0.0
with torch.no_grad():
    count = 0
    for test_inputs, test_labels in test_loader:
        
        test_outputs = best_model(test_inputs)
        if count == 0:
            example_True_3para = test_labels
            example_FromNN_3para = test_outputs
            count = 1
        
        test_loss_single = criterion(test_outputs, test_labels, test_inputs,mode = 2)
        test_loss += test_loss_single

        label_abs_err = torch.mean(torch.abs(test_labels-test_outputs))
        label_rel_err = torch.mean(torch.abs(test_labels-test_outputs)/test_outputs)
        label_abs_err_total += label_abs_err
        label_rel_err_total += label_rel_err 

        if test_loss_single>0.5 and count == 1:
            example_True_3para = test_inputs
            example_FromNN_3para = test_outputs
            count = 2 
            print(test_loss_single)    
        
    print(f'Test Loss: {test_loss / len(test_loader)}')
    print(f'三参数标签与输出的MAE: {label_abs_err_total  / len(test_loader)}')
    print(f'三参数标签与输出的MRE: {label_rel_err_total  / len(test_loader)}')

#模型在测试集跑出来的三参数计算出来的浓度曲线和真实三参数计算出来的浓度曲线对比
example_True_3para = (example_True_3para.cpu()).numpy()
print(example_True_3para)

example_FromNN_3para=example_FromNN_3para.cpu().numpy()
print(example_FromNN_3para)

print(f'Test Loss: {test_loss / len(test_loader)}')

id = 0
time = np.arange(0,75,0.005)

plasmaTrue,urineTrue,urinegTrue  =  BPS_BPTK_MultiParas(t = time,volunteer_ID =id, paras = example_True_3para ,mode = '63')
plasmaFromNN,urineFromNN,urinegFromNN  =  BPS_BPTK_MultiParas(t = time,volunteer_ID =id, paras = example_FromNN_3para ,mode = '63')


plt.subplot(221)
plt.plot(time,plasmaFromNN[0,:],label = 'FromNN_result')
plt.plot(time,plasmaTrue[0,:],label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()

plt.subplot(222)
plt.plot(time,urineFromNN[0,:],label = 'FromNN_result')
plt.plot(time,urineTrue[0,:],label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()

plt.subplot(223)
plt.plot(time,urinegFromNN[0,:],label = 'FromNN_result')
plt.plot(time,urinegTrue[0,:],label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()

plt.subplot(224)
plt.plot(time,plasmaFromNN[0,:],label = 'FromNN_result')
plt.plot(time,plasmaTrue[0,:],label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()

plt.show()

outputs = best_model(X_test)
X_test = inverse_transform(X_test, X_mean, X_std)
X_test = X_test.detach().numpy()
outputs = outputs.detach().numpy()

np.random.seed(0)
selected_indices = np.random.choice(X_test.shape[0], size=150, replace=False)
X_test = X_test[selected_indices,:]
outputs = outputs[selected_indices,:]

a = np.arange(0.5,5.1,0.5)
b = np.arange(6,15.1,1)
c = np.arange(18,42.1,6)
d = np.array([50,60,72])
sampling_time_range = np.hstack((a,b,c,d)) #采样时间节点，在0至75小时内共选取了28个时间节点 
sampling_time_index_plasma = (200*sampling_time_range).astype(int) #采样时间节点在求解器结果中的索引值
a = np.arange(1,15.1,2)
b = np.arange(24,42.1,6)
c = np.array([50,60,72])
sampling_time_range = np.hstack((a,b,c)) #采样时间节点，在0至75小时内共选取了15个时间节点 
sampling_time_index_urine = (200*sampling_time_range).astype(int) #采样时间节点在求解器结果中的索引值

plasma,urine,urineg  =  BPS_BPTK_MultiParas(t = time,volunteer_ID =id, paras = outputs ,mode = '63')
plasma = plasma[:,sampling_time_index_plasma]
urine = urine[:,sampling_time_index_urine]
urineg = urineg[:,sampling_time_index_urine]

FromNN = np.hstack((plasma,urine,urineg))



mse = np.mean((X_test- FromNN ) ** 2)
mre = np.mean(np.abs(X_test- FromNN)/np.maximum(X_test, 1E-9))
print(f'整个测试集的原始特征与输出三参数通过PBPK模型计算得到的对应采样点的MSE为  {mse}')
print(f'整个测试集的原始特征与输出三参数通过PBPK模型计算得到的对应采样点的MRE为  {mre}')
print(mre)