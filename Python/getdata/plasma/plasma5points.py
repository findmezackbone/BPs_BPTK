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

#j = 0
#time = np.arange(0,75,0.005) #七十五个小时的时间戳
#a = np.array([[15,5,5],[10,8,8],[20,10,6],[17,6,5]])
#data,_,_ = BPS_BPTK_MultiParas(t = time,volunteer_ID =1, paras = a ,mode = '63')

PlasmaFivePoints1 = np.zeros((np.shape(data)[0],6))
PlasmaFivePoints2 = np.zeros((np.shape(data)[0],8))

for i in tqdm(range(np.shape(data)[0])):
    less  = np.array(argrelextrema(data[i,:12000], np.less),dtype=int) #找极小值点，可能没有
    greater = np.array(argrelextrema(data[i,:], np.greater),dtype=int) #找极大值点,可能有两个
    half1 = np.abs(data[i,:greater[0,0]] - 0.5*data[i,greater[0,0]]).argmin() #找极大值点的1/2点，有两个
    #plt.plot(time,data[i,:],label = f'{i}')

    if less.size > 0:
        if data[i,greater[0,1]]>1/32*data[i,greater[0,0]] and data[i,greater[0,1]]<data[i,greater[0,0]]:
            half2 = np.abs(data[i,greater[0,0]:less[0,0]] - 0.5*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的1/2点，有两个
            PlasmaFivePoints1[i,:] = np.hstack((i,half1,greater[0,0],half2,less[0,0],greater[0,1])).astype(int)
            #plt.scatter((PlasmaFivePoints1[i,:]-1)/200,data[i,np.array(PlasmaFivePoints1[i,:],dtype=int)],c='red')
        elif data[i,greater[0,1]]<data[i,greater[0,0]]:
            half2 = np.abs(data[i,greater[0,0]:] - 0.5*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的1/2点，有两个
            quarter = np.abs(data[i,greater[0,0]:] - 0.25*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/4点
            eighth = np.abs(data[i,greater[0,0]:] - 0.125*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/8点
            sixteenth = np.abs(data[i,greater[0,0]:] - 1/16*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/16点
            thirtysecond = np.abs(data[i,greater[0,0]:] - 1/32*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/32点
            PlasmaFivePoints2[i,:] = np.hstack((i,half1,greater[0,0],half2,quarter ,eighth,sixteenth,thirtysecond)).astype(int)
    else:
        half2 = np.abs(data[i,greater[0,0]:] - 0.5*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的1/2点，有两个
        quarter = np.abs(data[i,greater[0,0]:] - 0.25*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/4点
        eighth = np.abs(data[i,greater[0,0]:] - 0.125*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/8点
        sixteenth = np.abs(data[i,greater[0,0]:] - 1/16*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/16点
        thirtysecond = np.abs(data[i,greater[0,0]:] - 1/32*data[i,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的右1/32点
        PlasmaFivePoints2[i,:] = np.hstack((i,half1,greater[0,0],half2,quarter ,eighth,sixteenth,thirtysecond)).astype(int)
        #plt.scatter((PlasmaFivePoints2[i,:]-1)/200,data[i,np.array(PlasmaFivePoints2[i,:],dtype=int)],c='blue')
       
PlasmaFivePoints1= PlasmaFivePoints1[[not np.all(PlasmaFivePoints1[i] == 0) for i in range(PlasmaFivePoints1.shape[0])], :]
PlasmaFivePoints2= PlasmaFivePoints2[[not np.all(PlasmaFivePoints2[i] == 0) for i in range(PlasmaFivePoints2.shape[0])], :]
 
print(np.shape(PlasmaFivePoints1))
print(np.shape(PlasmaFivePoints2))

np.save("Python\\getdata\\PlasmaFivePoints1.npy",PlasmaFivePoints1)
np.save("Python\\getdata\\PlasmaFivePoints2.npy",PlasmaFivePoints2)
#plt.legend()
#plt.show()

