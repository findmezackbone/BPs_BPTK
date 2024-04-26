import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import time
sys.path.append("Python") 
from BPS_init_function_MultiParas_copy import BPS_BPTK_MultiParas

from scipy import stats
j = 0
time1 = np.arange(0,75,0.005) #七十五个小时的时间戳
mean1 = 17.28
mean2 = 6.39
mean3 = 5.7

X1 = stats.truncnorm(-2, 2.3, loc=mean1, scale=5)
x1 = X1.rvs(size = 6,random_state = 43)
X2 = stats.truncnorm(-1, 1.3, loc=mean2, scale=5)
x2 = X2.rvs(size = 5,random_state = 43)
X3 = stats.truncnorm(-1, 1.2, loc=mean3, scale=5)
x3 = X3.rvs(size = 5,random_state = 43)

a = np.array(list(itertools.product(x1, x2, x3)))

start = time.perf_counter()

data = BPS_BPTK_MultiParas(t = time1,volunteer_ID =1, paras = a ,mode = '63')

end = time.perf_counter()

runTime = end - start
print("运行时间：", runTime)
paras = np.array(list(itertools.product(x1, x2, x3)))
#0.10864299066655803333333333333333s
#18.16208s