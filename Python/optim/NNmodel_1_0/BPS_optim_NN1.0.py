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


# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 准备数据
X = np.load("Python\optim\BPSplasma_init_Data.npy")  #输入数据

DSC_range = np.hstack((np.arange(15,10,-1.2),np.arange(15.5,20,0.5),np.arange(21,30,1.2),31,34,37))
PFO_range = np.hstack((1.2,np.arange(2,8,0.4),np.arange(8,15,1.8)))
u1_range = np.hstack((1.2,2.4,np.arange(3,8,0.4),np.arange(8.8,15,4/3)))
y = np.array(list(itertools.product(DSC_range, PFO_range, u1_range))) #标签

# 数据预处理 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.reshape(X, (10000, 68, 1))

X = torch.tensor(X).float()
y = torch.tensor(y).float()

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=20)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

# 将数据转移到 GPU（如果可用）
X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)

# 定义神经网络模型
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

# 初始化模型、损失函数和优化器
model = CustomLSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 准备 DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=32)

# 训练模型
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_labels)

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 保存效果最好的模型
        torch.save(model.state_dict(), 'Python\optim\model_best.pth')
    
    print(f'Epoch {epoch+1}, Training Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}')

# 加载效果最好的模型
best_model = CustomLSTM(input_size, hidden_size, num_layers, output_size).to(device)
best_model.load_state_dict(torch.load('Python\optim\model_best.pth'))

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
        count += 1
        test_loss += criterion(test_outputs, test_labels)

print(f'Test Loss: {test_loss / len(test_loader)}')


#一些对比
example_True_3para = (example_True_3para.cpu()).numpy()
example_True_3para = example_True_3para.flatten()
print(example_True_3para)


example_FromNN_3para=example_FromNN_3para.cpu().numpy()
example_FromNN_3para = example_FromNN_3para.flatten()
print(example_FromNN_3para)


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

plt.plot(time,abs(result_FromNN[:,25]/result_True[:,25]-1),label = 'Python_result')
plt.xlabel('time(h)')
plt.ylabel('Relative Error of TRUE and FROM-NN Results')
plt.show()