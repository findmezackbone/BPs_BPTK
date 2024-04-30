import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("Python") 
from BPS_init_function_MultiParas import BPS_BPTK_MultiParas
from scipy.signal import argrelextrema
from scipy import stats
from tqdm import tqdm 

data1 = np.load("Python\\optim\\DataFromBPTK\\huge\\urinebpsg_zzc.npy")
data2 = np.load("Python\\optim\\DataFromBPTK\\huge\\urinebpsg_SG.npy")

data = np.vstack((data1,data2))
data = np.diff(data)
print(np.shape(data))

index1 = np.load("Python\\getdata\\urineg\\UrinegFivePoints1.npy")
index2 = np.load("Python\\getdata\\urineg\\UrinegFivePoints2.npy")

UrinegFivePointsValue1 = np.zeros((np.shape(index1)[0],6))
UrinegFivePointsValue2 = np.zeros((np.shape(index1)[0],8))

for i in tqdm(range(np.shape(index1)[0])):
    UrinegFivePointsValue1[i,:] = np.hstack((index1[i,0],data[index1[i,0].astype(int), index1[i,1:].astype(int)]))

for i in tqdm(range(np.shape(index2)[0])):
    UrinegFivePointsValue2[i,:] = np.hstack((index2[i,0].astype(int),data[index2[i,0].astype(int), index2[i,1:].astype(int)]))
       
UrinegFivePointsValue1= UrinegFivePointsValue1[[not np.all(UrinegFivePointsValue1[i] == 0) for i in range(UrinegFivePointsValue1.shape[0])], :]
UrinegFivePointsValue2= UrinegFivePointsValue2[[not np.all(UrinegFivePointsValue2[i] == 0) for i in range(UrinegFivePointsValue2.shape[0])], :]
 
print(np.shape(UrinegFivePointsValue1))
print(np.shape(UrinegFivePointsValue2))

np.save("Python\\getdata\\urineg\\UrinegFivePointsValue1.npy",UrinegFivePointsValue1)
np.save("Python\\getdata\\urineg\\UrinegFivePointsValue2.npy",UrinegFivePointsValue2)
#plt.legend()
#plt.show()

