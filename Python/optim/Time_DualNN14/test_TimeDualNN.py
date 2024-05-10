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

def standard_transform_paras(x,mean,std):
    
    x = (x - mean) / std
    return x


def inverse_transform(x, mean, std):
    # 将标准化后的张量乘以标准差，然后加上均值
    x = x * std + mean
    return x

inputshape0 = 28
# 准备数据

X =  np.load("Python\optim\Time_DualNN14\database\input_add.npy")  
print(np.shape(X))

y = np.load("Python\optim\Time_DualNN14\database\label_add.npy")  
print(np.shape(y))
# 数据预处理 标准化数据

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 检查GPU是否可用
X = torch.tensor(X).float().to(device)
y = torch.tensor(y).float().to(device)

X,X_mean,X_std = standard_transform(X)

y,y_mean,y_std = standard_transform(y)
# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=20)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

# 将数据转移到 GPU（如果可用



X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)


X1 = np.load("Python\ForwardFitNN\WithTime\Database\ForwardInput_zzc.npy")
X2 = np.load("Python\ForwardFitNN\WithTime\Database\ForwardInput_SG.npy")
X_forward =  np.vstack((X1,X2))
X_forward  = torch.tensor(X_forward ).float().to(device)

_,X_forward_mean, X_forward_std = standard_transform(X_forward)
y1 = np.load("Python\ForwardFitNN\WithTime\Database\ForwardLabel_zzc.npy")  #输入标签数据
y2 = np.load("Python\ForwardFitNN\WithTime\Database\ForwardLabel_SG.npy")  #输入标签数据
y_forward = np.vstack((y1,y2))
y_forward  = torch.tensor(y_forward ).float().to(device)
_,y_forward_mean, y_forward_std = standard_transform(y_forward)

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
        self.input_dim = hyperparas['input_dim'] #15
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
hyperparas_reverse = {'input_dim':28,'hidden_dim':40,'hidden_nums':4,'output_dim':3,'block_layer_nums':3}

hyperparas_forward_urine = {'input_dim':4,'hidden_dim':50,'hidden_nums':5,'output_dim':2,'block_layer_nums':3}
learning_rate = 0.001
num_epochs = 300

batchsize_paras = {'train':512,'valid':256,'test':256}

# 初始化模型、损失函数和优化器


model_forward_urine = ResNN_Forward(hyperparas_forward_urine).to(device)
model_forward_urine.load_state_dict(torch.load('Python\\ForwardFitNN\\Settled_Model\\TimeFourTo2\\1\\model1.pth'))

loss_function=nn.MSELoss()

def criterion(output,label,input,batchsize,mode = 1):
    if mode == 1:
        return loss_function(output,label)

    

    if mode == 5:

        loss2 = 0.0
        #loss3 = 0.0
        input_inverse = inverse_transform(input, X_mean, X_std)
        '''    
        for i in range(int(inputshape0/2)):
            
            output_transformed = torch.cat((output, input_inverse[:,int(inputshape0)+i].view(-1,1)), dim=1)
            output_transformed = standard_transform_paras(output_transformed,X_forward_mean,X_forward_std)#把数据变得与正向模型的输入 匹配

            output_forward_urine = model_forward_urine(output_transformed) #模型输出的三参数代入至PBPK的拟合网络中得到一个代表15个尿液bps含量采样点的数组

            output_forward_urine = inverse_transform(output_forward_urine,y_forward_mean,y_forward_std) #把正向模型输出变正常
            output_forward_urine_add = torch.sum(output_forward_urine, dim=1)
            loss2 += loss_function(input_inverse[:,i],output_forward_urine_add)/(inputshape0/2)
            #loss3 += torch.mean(torch.abs(input[:,i] - output_forward_urine_add)/torch.max(input[:,i], torch.full_like(input[:,i], 1E-9)))
        '''    
        #以下不需要for循环
        output_inverse = inverse_transform(output, y_mean, y_std)
        labeltimes =output_inverse.repeat(int(inputshape0/2),1)

        output_transformed = torch.cat((labeltimes,input_inverse[:,int(inputshape0/2):inputshape0].t().reshape(-1,1)), dim=1)
        output_transformed = standard_transform_paras(output_transformed,X_forward_mean,X_forward_std)#把数据变得与正向模型的输入 匹配

        output_forward_urine = model_forward_urine(output_transformed) #模型输出的三参数代入至PBPK的拟合网络中得到一个代表15个尿液bps含量采样点的数组

        output_forward_urine_inverse = inverse_transform(output_forward_urine,y_forward_mean,y_forward_std) #把正向模型输出变正常
        output_forward_urine_add = torch.sum(output_forward_urine_inverse, dim=1)

        loss2 = loss_function(input_inverse[:,0:int(inputshape0/2)].t().reshape(int(inputshape0/2)*batchsize),output_forward_urine_add)
        loss1 = loss_function(output,label)

        
        
        loss1 = 1*1E2*loss1
        loss2 = 5*1E4*loss2
        #loss3 = loss3
        loss = loss1 + loss2 #+ loss3
        return loss,  loss1, loss2, 0







# 加载效果最好的模型
best_model = ResNN_Reverse(hyperparas_reverse).to(device)
best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_best.pth'))
#best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_pause1.pth'))
#best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_pause2.pth'))


# 在测试集上评估模型
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batchsize_paras['test'],shuffle = None)

test_loss = 0.0
with torch.no_grad():
    count = 0
    for test_inputs, test_labels in test_loader:
        
        test_outputs = best_model(test_inputs)
        if count == 0:
            example_True_3para = test_labels
            example_FromNN_3para = test_outputs
            count = 1
        
        test_loss_single,_,_,_ = criterion(test_outputs, test_labels,test_inputs,batchsize=len(test_inputs), mode =5)
        test_loss += test_loss_single/len(test_inputs)
 
        
    print(f'Test Loss: {test_loss / len(test_loader)}')


#模型在测试集跑出来的三参数计算出来的浓度曲线和真实三参数计算出来的浓度曲线对比
example_True_3para = inverse_transform(example_True_3para, y_mean, y_std)
example_FromNN_3para = inverse_transform(example_FromNN_3para, y_mean, y_std)

example_True_3para = example_True_3para.cpu().numpy()
#print(example_True_3para)

example_FromNN_3para=example_FromNN_3para.cpu().numpy()
#print(example_FromNN_3para)

print(f'Test Loss: {test_loss / len(test_loader)}')

id = 0
time = np.arange(0,75,0.005)

_,urineTrue,urinegTrue  =  BPS_BPTK_MultiParas(t = time,volunteer_ID =id, paras = example_True_3para ,mode = '63')
_,urineFromNN,urinegFromNN  =  BPS_BPTK_MultiParas(t = time,volunteer_ID =id, paras = example_FromNN_3para ,mode = '63')

test_id = 200
print(example_True_3para[test_id,:])
print(example_FromNN_3para[test_id,:])

plt.subplot(121)
plt.plot(time,urineFromNN[test_id,:],label = 'FromNN_result')
plt.plot(time,urineTrue[test_id,:],label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()

plt.subplot(122)
plt.plot(time,urinegFromNN[test_id,:],label = 'FromNN_result')
plt.plot(time,urinegTrue[test_id,:],label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPSg in plasma')
plt.legend()


plt.show()

  