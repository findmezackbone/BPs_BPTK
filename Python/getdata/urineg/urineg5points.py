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
#print(np.shape(data)[0])

#j = 0
#time = np.arange(0,75,0.005) #七十五个小时的时间戳
#a = np.array([[15,5,5],[10,8,8],[22,10,6],[6,6,5]])
#_,data,_ = BPS_BPTK_MultiParas(t = time,volunteer_ID =1, paras = a ,mode = '63')

data = np.diff(data)
UrinegFivePoints1 = np.zeros((np.shape(data)[0],6))
UrinegFivePoints2 = np.zeros((np.shape(data)[0],8))

#time = time[:-1]

for i in tqdm(range(np.shape(data)[0])):
    less  = np.array(argrelextrema(data[i,:12000], np.less),dtype=int) #找极小值点，可能没有
    greater = np.array(argrelextrema(data[i,:], np.greater),dtype=int) #找极大值点,可能有两个
    half1 = np.abs(data[i,:greater[0,0]] - 0.5*data[i,greater[0,0]]).argmin() #找极大值点的1/2点，有两个
    #plt.plot(time,data[i,:],label = f'{i}')

    if less.size > 0:
        if data[i,greater[0,1]]>1/32*data[i,greater[0,0]] and data[i,greater[0,1]]<data[i,greater[0,0]]:
            half2 = np.abs(data[i,greater[0,0]:less[0,0]] - 0.5*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的1/2点，有两个
            UrinegFivePoints1[i,:] = np.hstack((i,half1,greater[0,0],half2,less[0,0],greater[0,1])).astype(int)
            #plt.scatter((UrinegFivePoints1[i,:]-1)/200,data[i,np.array(UrinegFivePoints1[i,:],dtype=int)],c='red')
        elif data[i,greater[0,1]]<data[i,greater[0,0]]:
            half2 = np.abs(data[i,greater[0,0]:] - 0.5*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的1/2点，有两个
            quarter = np.abs(data[i,greater[0,0]:] - 0.25*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/4点
            eighth = np.abs(data[i,greater[0,0]:] - 0.125*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/8点
            sixteenth = np.abs(data[i,greater[0,0]:] - 1/16*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/16点
            thirtysecond = np.abs(data[i,greater[0,0]:] - 1/32*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/32点
            UrinegFivePoints2[i,:] = np.hstack((i,half1,greater[0,0],half2,quarter ,eighth,sixteenth,thirtysecond)).astype(int)
            #plt.scatter((UrinegFivePoints2[i,:]-1)/200,data[i,np.array(UrinegFivePoints2[i,:],dtype=int)],c='blue')
    else:
        half2 = np.abs(data[i,greater[0,0]:] - 0.5*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的1/2点，有两个
        quarter = np.abs(data[i,greater[0,0]:] - 0.25*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/4点
        eighth = np.abs(data[i,greater[0,0]:] - 0.125*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/8点
        sixteenth = np.abs(data[i,greater[0,0]:] - 1/16*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/16点
        thirtysecond = np.abs(data[i,greater[0,0]:] - 1/32*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/32点
        UrinegFivePoints2[i,:] = np.hstack((i,half1,greater[0,0],half2,quarter ,eighth,sixteenth,thirtysecond)).astype(int)
        #plt.scatter((UrinegFivePoints2[i,:]-1)/200,data[i,np.array(UrinegFivePoints2[i,:],dtype=int)],c='blue')
       
UrinegFivePoints1= UrinegFivePoints1[[not np.all(UrinegFivePoints1[i] == 0) for i in range(UrinegFivePoints1.shape[0])], :]
UrinegFivePoints2= UrinegFivePoints2[[not np.all(UrinegFivePoints2[i] == 0) for i in range(UrinegFivePoints2.shape[0])], :]
 
print(np.shape(UrinegFivePoints1))
print(np.shape(UrinegFivePoints2))

np.save("Python\\getdata\\urineg\\UrinegFivePoints1.npy",UrinegFivePoints1)
np.save("Python\\getdata\\urineg\\UrinegFivePoints2.npy",UrinegFivePoints2)
#plt.legend()
#plt.show()

