import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema
from scipy import stats

a = np.array([[1,2,3,4],[2,4,6,8]])
b = np.diff(a)

print(b)