import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
#plt.rcParams['text.usetex'] = True
import sys
sys.path.append("Python") 
from BPS_init_function_MultiParas import BPS_BPTK_MultiParas
from scipy.signal import argrelextrema
from scipy import stats
a = np.arange(1,15.1,2)
b = np.arange(24,42.1,6)
c = np.array([50,60,72])
sampling_time_range_urine = np.hstack((a,b,c)) #采样时间节点，在0至75小时内共选取了15个时间节点 
sampling_time_index_urine = (200*sampling_time_range_urine).astype(int) #采样时间节点在求解器结果中的索引值
sampling_time_index_urine = np.array([101,433,608,1653,3661])
sampling_time_range_urine = sampling_time_index_urine/200
a = np.arange(0.5,5.1,0.5)
b = np.arange(6,15.1,1)
c = np.arange(18,42.1,6)
d = np.array([50,60,72])
sampling_time_range_plasma = np.hstack((a,b,c,d)) #采样时间节点，在0至75小时内共选取了28个时间节点 
sampling_time_index_plasma = (200*sampling_time_range_plasma).astype(int) #采样时间节点在求解器结果中的索引值
sampling_time_index_plasma = np.array([98,433,601,1644,3651])
sampling_time_range_plasma = sampling_time_index_plasma/200

j = 0
time = np.arange(0,75,0.005) #七十五个小时的时间戳
a = np.array([[14  ,   6.39  , 6]])
data,urinebps,urinebpsg = BPS_BPTK_MultiParas(t = time,volunteer_ID =j, paras = a ,mode = '63')

#画血浆内bps含量的曲线
plt.plot(time,data[0,:],label = '血浆BPS含量')
less  = np.array(argrelextrema(data[0,:], np.less),dtype=int)
greater = np.array(argrelextrema(data[0,:], np.greater),dtype=int)
print(data[0,greater])
half1 = np.abs(data[0,:greater[0,0]] - 0.5*data[0,greater[0,0] ]).argmin()
if less.size > 0:
    half2 = np.abs(data[0,greater[0,0]:less[0,0]] - 0.5*data[0,greater[0,0]]).argmin()
    half2 = half2 + greater[0,0]
    thirtysecond =  np.abs(data[0,greater[0,0]] - 1/32*data[0,greater[0,0]]).argmin()+ greater[0,0]
    #plt.scatter(less/200,data[0,less],c='red')


else:
    half2 = np.abs(data[0,greater[0,0]:] - 0.5*data[0,greater[0,0]]).argmin()+ greater[0,0]#找极大值点的1/2点，有两个
    #plt.scatter((PlasmaFivePoints2[i,:]-1)/200,data[i,np.array(PlasmaFivePoints2[i,:],dtype=int)],c='blue')
    thirtysecond =  np.abs(data[0,greater[0,0]:]- 1/32*data[0,greater[0,0]]).argmin()+ greater[0,0]

plt.annotate(r'$\alpha_1$='f'{a[0,0]}, 'r'$\alpha_2$='f'{a[0,1]}, 'r'$\alpha_3$='f'{a[0,2]}',xy = [0.4,0.6], xycoords='figure fraction',weight='bold',color = 'blue', fontsize = 14)

#plt.annotate(f'alpha_1={a[0,0]} alpha_2={a[0,1]}  alpha_3={a[0,2]}',xy = [0.3,0.6], xycoords='figure fraction',weight='bold',color = 'blue', fontsize = 14)
#plt.scatter(greater/200,data[0,greater],c='red')
#plt.scatter(half1/200,data[0,half1 ],c='red')
#plt.scatter(half2/200,data[0,half2 ],c='red')
#plt.scatter(thirtysecond/200,data[0,thirtysecond],c='red')
plt.scatter(sampling_time_range_plasma,data[0,sampling_time_index_plasma],c='red', label='采样点')
plt.legend()

plt.show()

plt.annotate(r'$\alpha_1$='f'{a[0,0]}, 'r'$\alpha_2$='f'{a[0,1]}, 'r'$\alpha_3$='f'{a[0,2]}',xy = [0.4,0.6], xycoords='figure fraction',weight='bold',color = 'blue', fontsize = 14)
plt.plot(time,urinebps[0,:],label = '尿液BPS含量')
plt.plot(time,urinebpsg[0,:],label = '尿液BPS-g含量')
plt.scatter(sampling_time_range_urine,urinebps[0,sampling_time_index_urine],c='red', label='BPS采样点')
plt.scatter(sampling_time_range_urine,urinebpsg[0,sampling_time_index_urine],c='green', label='BPS-g采样点')
plt.legend()
plt.show()