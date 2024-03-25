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

# 将数据转移到 GPU（如果可用

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)

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
num_epochs = 450
dropout_prob = 0

# 初始化模型、损失函数和优化器
model = CustomLSTM(input_size, hidden_size1, num_layers, output_size, dropout_prob).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 准备 DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=32)

patience_counter = 0
patience_on  = 0
patience = 10
stop_training = 0
# 热键函数
def on_press(key):
    global patience_on
    global stop_training
    if key.name == 's':#终止训练，储存最好模型
        print("Training stopped. Saving current best model...")
        print(f'best validation loss : {best_val_loss/ len(val_loader)}')
        best_model = CustomLSTM(input_size, hidden_size1, num_layers, output_size, dropout_prob).to(device)
        best_model.load_state_dict(torch.load('Python\optim\model_best.pth'))
        # 保存效果最好的模型
        torch.save(best_model.state_dict(), 'Python\optim\model_best.pth')
        stop_training = 1

    if key.name == 'q': #中途储存当前最好模型，但并不终止训练
        print("Saving current best model to pause1...")
        print(f'best validation loss : {best_val_loss/ len(val_loader)}')
        best_model = CustomLSTM(input_size, hidden_size1, num_layers, output_size, dropout_prob).to(device)
        best_model.load_state_dict(torch.load('Python\optim\model_best.pth'))
        # 保存效果最好的模型
        torch.save(best_model.state_dict(), 'Python\optim\model_pause1.pth')
        
    if key.name == 'w': #中途储存当前最好模型，但并不终止训练
        print("Saving current best model to pause2...")
        print(f'best validation loss : {best_val_loss/ len(val_loader)}')
        best_model = CustomLSTM(input_size, hidden_size1, num_layers, output_size, dropout_prob).to(device)
        best_model.load_state_dict(torch.load('Python\optim\model_best.pth'))
        # 保存效果最好的模型
        torch.save(best_model.state_dict(), 'Python\optim\model_pause2.pth')

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
        train_loss += loss.item()

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_labels)

    if val_loss / len(val_loader) < 0.75:
        patience_on = 1
    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存效果最好的模型
        torch.save(model.state_dict(), 'Python\optim\model_best.pth')
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
best_model = CustomLSTM(input_size, hidden_size1, num_layers, output_size, dropout_prob).to(device)
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
        count = 1

        test_loss += criterion(test_outputs, test_labels)

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

plt.plot(time,abs(result_FromNN[:,25]/(result_True[:,25]+1E-30)-1))
plt.xlabel('time(h)')
plt.ylabel('Relative Error of TRUE and FROM-NN Results')
plt.show()