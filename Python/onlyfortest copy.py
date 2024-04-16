import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("Python") 
from BPS_init_function_MultiParas import BPS_BPTK_MultiParas

from scipy import stats

j = 0
time = np.arange(0,75,0.005) #七十五个小时的时间戳
a = np.array([[16,5,5]])
blood,bps,bpsg = BPS_BPTK_MultiParas(t = time,volunteer_ID =1, paras = a ,mode = '63')

plt.plot(time,bps[0],label = 'bps')

plt.plot(time,bpsg[0],label = 'bpsg')

plt.legend()

plt.show()

plt.plot(time,blood[0]*1000,label = 'blood')
plt.legend()

plt.show()