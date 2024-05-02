import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import time
sys.path.append("Python") 
from BPS_init_function_MultiParas_copy import BPS_BPTK_MultiParas
from scipy import stats
true_index = np.loadtxt(open("Documents\Volunteer_data.csv"),delimiter=",",skiprows=1,usecols=[1,2,3,4]) #真实受试者的尿液数据

true_index =true_index[0:17,0]

timeline = np.arange(0,75,0.005)
a = np.ones(np.shape(timeline)).flatten()

plt.plot(timeline, a)

plt.scatter(true_index, np.ones(np.shape(true_index)).flatten() )
plt.show()
j = 0
time1 = np.arange(0,75,0.005) #七十五个小时的时间戳
mean1 = 17.28
mean2 = 6.39
mean3 = 5.7

