import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("Python") 
from BPS_init_function import BPS_BPTK 
from scipy.optimize import minimize,dual_annealing,basinhopping,differential_evolution
from hyperopt import fmin, tpe, hp
MWBPS = 250.27  #BPS的摩尔质量，转化暴露量ng为mol用
time = np.arange(0, 75, 0.005) #七十五个小时的时间戳
para_BP_individual = np.loadtxt(open("Documents\para_BP_individual.csv"),delimiter=",",skiprows=1,usecols=[2,3,4,5]) #受试者的生理参数
Volunteer_data = np.loadtxt(open("Documents\Volunteer_data.csv"),delimiter=",",skiprows=1,usecols=[1,2,3,4]) #真实受试者的尿液数据
#1至17是第一位（17）；18至29是第二位（12）；30至44是第三位（15）；45至55是第四位（11）

Volunteer_data[:,0]=np.round(Volunteer_data[:,0],2) #将真实数据的时间戳四舍五入为二位小数
t_data=Volunteer_data[:,0] #给受试者采尿的时间节点序列
c_data=Volunteer_data[:,2]/Volunteer_data[:,3]*1000*1000 #总BPS:BPS+BPSg
f_data=Volunteer_data[:,1]/Volunteer_data[:,3]*1000*1000 #未结合的BPS

urine_init_per=np.zeros((55,8)) #提前分配内存,用于储存模型计算出来的对应采尿时间节点的尿液数据
urine_out_per=np.zeros((15001,8)) 

theta_s1=np.array([35.278691680713024,6.01017307502789,13.980673126293965]) #给待优化参数一个初值

for j in range(0,4):
    DOSE_d = para_BP_individual[17,j]*para_BP_individual[16,j]/MWBPS  
    result = BPS_BPTK(t = time,volunteer_ID =j, DSC_0 = theta_s1[0], PFO_0 = theta_s1[1], u1_0 = theta_s1[2] ,mode = '63')
    total_u=result[:,37]+result[:,54] #前者是尿液中BPS，后者是尿液中BPS-g
    t_u=total_u/(DOSE_d)*1000*1000 #尿液中BPS＋BPSg的总量的时间序列
    f_u=result[:,37]/(DOSE_d)*1000*1000 #尿液中未结合BPS的量的时间序列
    index = t_data/0.005+1
    index = index .astype('int64')
    
    urine_init_per[:,j]=t_u[index]
    urine_init_per[:,j+4]=f_u[index]
    
SE1 =np.sum(((urine_init_per[0:17,0] - c_data[0:17])/142.853614 )** 2 )
SE2 = np.sum(((urine_init_per[17:29,1]- c_data[17:29])/ 96.037827)** 2  )
SE3 = np.sum(((urine_init_per[29:44,2]- c_data[29:44])/ 91.694228)** 2  )
SE4 =  np.sum(((urine_init_per[44:55,3]- c_data[44:55])/135.018903)** 2)

SE5 =np.sum(((urine_init_per[0:17,4] - f_data[0:17])/ 10.240965)** 2 )
SE6 = np.sum(((urine_init_per[17:29,5]- f_data[17:29])/ 10.590446)** 2 )
SE7 =np.sum(( (urine_init_per[29:44,6]- f_data[29:44])/ 13.855707 )** 2  )
SE8 = np.sum(( (urine_init_per[44:55,7]- f_data[44:55])/ 10.644076 )** 2)
SSE = SE5+SE6+SE7+SE8+SE1+SE2+SE3+SE4

print(SSE)