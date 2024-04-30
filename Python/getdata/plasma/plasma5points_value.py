import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("Python") 
from BPS_init_function_MultiParas import BPS_BPTK_MultiParas
from scipy.signal import argrelextrema
from scipy import stats
from tqdm import tqdm 

data1 = np.load("Python\optim\DataFromBPTK\huge\plasma_zzc.npy")
data2 = np.load("Python\optim\DataFromBPTK\huge\plasma_SG.npy")

data = np.vstack((data1,data2))
print(np.shape(data)[0])

index1 = np.load("Python\getdata\PlasmaFivePoints1.npy")
index2 = np.load("Python\getdata\PlasmaFivePoints2.npy")

PlasmaFivePointsValue1 = np.zeros((np.shape(index1)[0],6))
PlasmaFivePointsValue2 = np.zeros((np.shape(index1)[0],8))

for i in tqdm(range(np.shape(index1)[0])):
    PlasmaFivePointsValue1[i,:] = np.hstack((index1[i,0],data[index1[i,0].astype(int), index1[i,1:].astype(int)]))

for i in tqdm(range(np.shape(index2)[0])):
    PlasmaFivePointsValue2[i,:] = np.hstack((index2[i,0].astype(int),data[index2[i,0].astype(int), index2[i,1:].astype(int)]))
       
PlasmaFivePointsValue1= PlasmaFivePointsValue1[[not np.all(PlasmaFivePointsValue1[i] == 0) for i in range(PlasmaFivePointsValue1.shape[0])], :]
PlasmaFivePointsValue2= PlasmaFivePointsValue2[[not np.all(PlasmaFivePointsValue2[i] == 0) for i in range(PlasmaFivePointsValue2.shape[0])], :]
 
print(np.shape(PlasmaFivePointsValue1))
print(np.shape(PlasmaFivePointsValue2))

np.save("Python\\getdata\\PlasmaFivePointsValue1.npy",PlasmaFivePointsValue1)
np.save("Python\\getdata\\PlasmaFivePointsValue2.npy",PlasmaFivePointsValue2)
#plt.legend()
#plt.show()

