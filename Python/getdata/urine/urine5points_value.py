import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("Python") 
from BPS_init_function_MultiParas import BPS_BPTK_MultiParas
from scipy.signal import argrelextrema
from scipy import stats
from tqdm import tqdm 

data1 = np.load("Python\\optim\\DataFromBPTK\\huge\\urinebps_zzc.npy")
data2 = np.load("Python\\optim\\DataFromBPTK\\huge\\urinebps_SG.npy")

data = np.vstack((data1,data2))
data = np.diff(data)
print(np.shape(data)[0])

index1 = np.load("Python\\getdata\\urine\\UrineFivePoints1.npy")
index2 = np.load("Python\\getdata\\urine\\UrineFivePoints2.npy")

UrineFivePointsValue1 = np.zeros((np.shape(index1)[0],6))
UrineFivePointsValue2 = np.zeros((np.shape(index1)[0],8))

for i in tqdm(range(np.shape(index1)[0])):
    UrineFivePointsValue1[i,:] = np.hstack((index1[i,0],data[index1[i,0].astype(int), index1[i,1:].astype(int)]))

for i in tqdm(range(np.shape(index2)[0])):
    UrineFivePointsValue2[i,:] = np.hstack((index2[i,0].astype(int),data[index2[i,0].astype(int), index2[i,1:].astype(int)]))
       
UrineFivePointsValue1= UrineFivePointsValue1[[not np.all(UrineFivePointsValue1[i] == 0) for i in range(UrineFivePointsValue1.shape[0])], :]
UrineFivePointsValue2= UrineFivePointsValue2[[not np.all(UrineFivePointsValue2[i] == 0) for i in range(UrineFivePointsValue2.shape[0])], :]
 
print(np.shape(UrineFivePointsValue1))
print(np.shape(UrineFivePointsValue2))

np.save("Python\\getdata\\urine\\UrineFivePointsValue1.npy",UrineFivePointsValue1)
np.save("Python\\getdata\\urine\\UrineFivePointsValue2.npy",UrineFivePointsValue2)
#plt.legend()
#plt.show()

