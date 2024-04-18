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
import keyboard


# 准备数据
X = np.load("Python\optim\DataFromBPTK\plasma68\plasma68_zzc.npy")  #输入数据
print(np.shape(X))
#DSC_range = np.hstack((np.arange(15,10,-1.2),np.arange(15.5,20,0.5),np.arange(21,30,1.2),31,34,37))
#PFO_range = np.hstack((1.2,np.arange(2,8,0.4),np.arange(8,15,1.8)))
#u1_range = np.hstack((1.2,2.4,np.arange(3,8,0.4),np.arange(8.8,15,4/3)))
#y1 = np.array(list(itertools.product(DSC_range, PFO_range, u1_range))) #标签#

y2 = np.load("Python\optim\DataFromBPTK\labels_zzc.npy")
y = y2
#y3 = np.load("Python\optim\DataFromBPTK\labels_SG.npy")
#y =  np.vstack((y1,y2,y3))
# 数据预处理 标准化数据

# 数据预处理 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)



X = torch.tensor(X).float()
y = torch.tensor(y).float()

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.15, random_state=20)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

# 将数据转移到 GPU（如果可用

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)



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
num_epochs = 300


# 初始化模型、损失函数和优化器
model = CustomResNN(hyperparas).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 准备 DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=32)

patience_counter = 0
patience_on  = 0
patience = 9
stop_training = 0
# 热键函数
def on_press(key):
    global patience_on
    global stop_training
    if key.name == 's':#终止训练，储存最好模型
        print("Training stopped. Saving current best model...")
        print(f'best validation loss : {best_val_loss/ len(val_loader)}')
        best_model = CustomResNN(hyperparas).to(device)
        best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_best.pth'))
        # 保存效果最好的模型
        torch.save(best_model.state_dict(), 'Python\optim\Temporary_Model\model_best.pth')
        stop_training = 1

    if key.name == 'q': #中途储存当前最好模型，但并不终止训练
        print("Saving current best model to pause1...")
        print(f'best validation loss : {best_val_loss/ len(val_loader)}')
        best_model = CustomResNN(hyperparas).to(device)
        best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_best.pth'))
        # 保存效果最好的模型
        torch.save(best_model.state_dict(), 'Python\optim\Temporary_Model\model_pause1.pth')
        
    if key.name == 'w': #中途储存当前最好模型，但并不终止训练
        print("Saving current best model to pause2...")
        print(f'best validation loss : {best_val_loss/ len(val_loader)}')
        best_model = CustomResNN(hyperparas).to(device)
        best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_best.pth'))
        # 保存效果最好的模型
        torch.save(best_model.state_dict(), 'Python\optim\Temporary_Model\model_pause2.pth')

    if key.name == 'e': #中途储存当前最好模型，但并不终止训练
        print("Saving current best model to pause3...")
        print(f'best validation loss : {best_val_loss/ len(val_loader)}')
        best_model = CustomResNN(hyperparas).to(device)
        best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_best.pth'))
        # 保存效果最好的模型
        torch.save(best_model.state_dict(), 'Python\optim\Temporary_Model\model_pause3.pth')

    if key.name == 'o': #开启early stopping
        patience_on  = 1
        print("early stopping is turned on")
        
    
keyboard.on_press(on_press)
# 训练模型
best_val_loss = float('inf')
for epoch in range(num_epochs):
    if stop_training:
        break
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += (loss/len(inputs)).item()

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_labels)/len(val_inputs)

    if val_loss / len(val_loader) <0.004:
        patience_on = 1
    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存效果最好的模型
        torch.save(model.state_dict(), 'Python\optim\Temporary_Model\model_best.pth')
    else:
        if patience_on == 1:
            patience_counter += 1
    
    if patience_counter >= patience:
        print(f'在第{epoch+1}个epoch处，Validation loss did not improve for {patience} epochs. Early stopping...')
        print(f'best validation loss : {best_val_loss/ len(val_loader)}')
        break
    if patience_on == 0:
        print(f'Epoch {epoch+1}, Training Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}, best val-loss now : {best_val_loss / len(val_loader)}')
    else: 
        print(f'Epoch {epoch+1}, Training Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}, earlystopping is on, {patience_counter}steps after last bestloss, best val-loss now : {best_val_loss / len(val_loader)}') 


keyboard.unhook_all()
# 加载效果最好的模型
best_model = CustomResNN(hyperparas).to(device)
best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_best.pth'))

# 在测试集上评估模型
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1,shuffle = None)

test_loss = 0.0
with torch.no_grad():
    count = 0
    for test_inputs, test_labels in test_loader:
        
        test_outputs = best_model(test_inputs)
        if count == 0:
            example_True_3para = test_labels
            example_FromNN_3para = test_outputs
        count = 1

        test_loss += criterion(test_outputs, test_labels)/len(test_inputs)

    print(f'Test Loss: {test_loss / len(test_loader)}')


#模型在测试集跑出来的三参数计算出来的浓度曲线和真实三参数计算出来的浓度曲线对比
example_True_3para = (example_True_3para.cpu()).numpy()
example_True_3para = example_True_3para.flatten()
print(example_True_3para)


example_FromNN_3para=example_FromNN_3para.cpu().numpy()
example_FromNN_3para = example_FromNN_3para.flatten()
print(example_FromNN_3para)

print(f'Test Loss: {test_loss / len(test_loader)}')

id = 0
time = np.arange(0,75,0.005)

result_FromNN = BPS_BPTK(t = time,volunteer_ID =id, DSC_0=example_FromNN_3para[0], PFO_0=example_FromNN_3para[1], u1_0=example_FromNN_3para[2] ,mode = '63')
result_True = BPS_BPTK(t = time,volunteer_ID =id, DSC_0=example_True_3para[0], PFO_0=example_True_3para[1], u1_0=example_True_3para[2] ,mode = '63')

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

plt.plot(time[1:],abs(result_FromNN[1:,25]/(result_True[1:,25]+1E-20)-1))
plt.xlabel('time(h)')
plt.ylabel('Relative Error of TRUE and FROM-NN Results')
plt.show()