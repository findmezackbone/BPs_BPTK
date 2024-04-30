import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("Python") 
from BPS_init_function_MultiParas import BPS_BPTK_MultiParas
from scipy.signal import argrelextrema
from scipy import stats

j = 0
time = np.arange(0,75,0.005) #七十五个小时的时间戳
a = np.array([[21.5  ,   6.39  ,  6.2625]])
data,urinebps,urinebpsg = BPS_BPTK_MultiParas(t = time,volunteer_ID =j, paras = a ,mode = '63')

plt.plot(time,data[0,:],label = 'blood')
less  = np.array(argrelextrema(data[0,:], np.less),dtype=int)
greater = np.array(argrelextrema(data[0,:], np.greater),dtype=int)
print(data[0,greater])
half1 = np.abs(data[0,:greater[0,0]] - 0.5*data[0,greater[0,0] ]).argmin()
if less.size > 0:
    half2 = np.abs(data[0,greater[0,0]:less[0,0]] - 0.5*data[0,greater[0,0]]).argmin()
    half2 = half2 + greater[0,0]
    thirtysecond =  np.abs(data[0,greater[0,0]] - 1/32*data[0,greater[0,0]]).argmin()+ greater[0,0]
    plt.scatter(less/200,data[0,less],c='red')


else:
    half2 = np.abs(data[0,greater[0,0]:] - 0.5*data[0,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的1/2点，有两个
    #plt.scatter((PlasmaFivePoints2[i,:]-1)/200,data[i,np.array(PlasmaFivePoints2[i,:],dtype=int)],c='blue')
    thirtysecond =  np.abs(data[0,greater[0,0]:]- 1/32*data[0,greater[0,0]]).argmin()+ greater[0,0]

plt.annotate(f'DSC={a[0,0]}  PFO={a[0,1]}  u1={a[0,2]}',xy = [0.3,0.6], xycoords='figure fraction',weight='bold',color = 'blue', fontsize = 14)
plt.scatter(greater/200,data[0,greater],c='red')
plt.scatter(half1/200,data[0,half1 ],c='red')
plt.scatter(half2/200,data[0,half2 ],c='red')
plt.scatter(thirtysecond/200,data[0,thirtysecond],c='red')
plt.legend()

plt.show()


plt.plot(time,urinebps[0,:],label = 'bps')
plt.plot(time,urinebpsg[0,:],label = 'bpsg')

plt.legend()
plt.show()

