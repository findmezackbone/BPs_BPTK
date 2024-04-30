import numpy as np
import pandas as pd

# 加载.npy文件
array = np.load('Python\\getdata\\urine\\UrineFivePointsValue1.npy')

# 将NumPy数组转换为Pandas DataFrame
df = pd.DataFrame(array)

# 将DataFrame保存为CSV文件
df.to_csv('Python\\getdata\\urine\\UrineFivePointsValue1.csv', index=False)
