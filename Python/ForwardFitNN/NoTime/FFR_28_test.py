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
import keyboard
from sklearn.metrics import r2_score

# 准备数据
y1 = np.load("Python\optim\DataFromBPTK\plasma28\plasma28_zzc.npy")  #输入标签数据
y2 = np.load("Python\optim\DataFromBPTK\plasma28\plasma28_SG.npy")  #输入标签数据


X1 = np.load("Python\optim\DataFromBPTK\labels_zzc.npy")
X2 = np.load("Python\optim\DataFromBPTK\labels_SG.npy")

y =  np.vstack((y1,y2))
X =  np.vstack((X1,X2))


# 数据预处理 标准化数据
def label_transform(x): #预处理标签数据
    x =  np.log(10000*x+1) #对数转换
    return(x)

def label_transform_reverse(x): #翻转预处理
    x = (np.exp(x)-1 ) /10000
    return(x)

scaler = StandardScaler()
#X = scaler.fit_transform(X)

y = label_transform(y)  

X = torch.tensor(X).float()
y = torch.tensor(y).float()

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.12, random_state=20)
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
    

class CustomResNN(nn.Module):
    def __init__(self,hyperparas):
        super().__init__()
        self.input_dim = hyperparas['input_dim'] #3
        self.hidden_dim = hyperparas['hidden_dim'] #30
        self.hidden_nums = hyperparas['hidden_nums'] #3
        self.output_dim = hyperparas['output_dim'] #28
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
hyperparas = {'input_dim':3,'hidden_dim':30,'hidden_nums':3,'output_dim':28,'block_layer_nums':3}

#hyperparas = {'input_dim':3,'hidden_dim':64,'hidden_nums':10,'output_dim':28,'block_layer_nums':3}
learning_rate = 0.001
num_epochs = 300

# 初始化模型、损失函数和优化器
criterion = nn.MSELoss()

# 准备 DataLoader

# 加载效果最好的模型
best_model = CustomResNN(hyperparas)
#best_model.load_state_dict(torch.load('Python\ForwardFitNN\Temporary_Model\model_best.pth'))
#best_model.load_state_dict(torch.load('Python\ForwardFitNN\Temporary_Model\model_pause3.pth'))
best_model.load_state_dict(torch.load('Python\ForwardFitNN\Settled_Model\\threeTo28\\model1.pth'))

# 在测试集上评估模型
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1,shuffle = None)

test_loss = 0.0
test_relativeerror = 0.0
with torch.no_grad():
    count = 0
    for test_inputs, test_labels in test_loader:
        
        test_outputs = best_model(test_inputs)
        if count == 0:
            example_paras = test_inputs
            FromNN = test_outputs
            count = 1
        
        test_loss += criterion(test_outputs, test_labels)
        
        RelErr = torch.mean(torch.abs(test_outputs - test_labels)/torch.max(test_labels, torch.full_like(test_labels, 1E-7)))
        
        if RelErr>300 and count == 1:
            example_paras = test_inputs
            FromNN = test_outputs
            count = 2 

            print(RelErr)
            
            
            
              

        test_relativeerror += RelErr

    print(f'Test Loss: {test_loss / len(test_loader)}')
    print(f'测试集上标签与输出的MRE: {test_relativeerror / len(test_loader)}')

#模型在测试集跑出来的三参数计算出来的浓度曲线和真实三参数计算出来的浓度曲线对比
FromNN_result = FromNN.cpu().numpy().flatten()
FromNN_result = label_transform_reverse(FromNN_result)
example_paras =  example_paras.cpu().numpy().flatten()
print(example_paras)
id = 0
time = np.arange(0,75,0.005)
result_True = BPS_BPTK(t = time,volunteer_ID =id, DSC_0=example_paras [0], PFO_0=example_paras [1], u1_0=example_paras [2] ,mode = '63')

a = np.arange(0.5,5.1,0.5)
b = np.arange(6,15.1,1)
c = np.arange(18,42.1,6)
d = np.array([50,60,72])
sampling_time_range = np.hstack((a,b,c,d)) #采样时间节点，在0至75小时内共选取了68个时间节点 
sampling_time_index = (200*sampling_time_range).astype(int) #采样时间节点在求解器结果中的索引值

plt.plot(time,result_True[:,25],label = '真实曲线')
plt.scatter(sampling_time_range,FromNN_result,label = '网络输出',c='red')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()
plt.show()

outputs = best_model(X_test)
y_test = y_test.detach().numpy()
outputs = outputs.detach().numpy()
y_test = label_transform_reverse(y_test)
outputs = label_transform_reverse(outputs)
mse = np.mean((y_test - outputs ) ** 2)
mre = np.mean(np.abs(y_test - outputs )/np.maximum(y_test, 1E-7))
R2 =  r2_score( y_test,outputs) #决定系数
print(f'整个测试集的原始标签与输出的真实变换的MSE为  {mse}')
print(f'整个测试集的原始标签与输出的真实变换的MRE为  {mre}')
print(f'整个测试集的原始标签与输出的真实变换的决定系数为  {R2}')