import torch
import numpy as np

mean_abs_err =2
mean_rel_err =3
mean_err = np.array([0,0])
mean_err[0] = mean_abs_err
mean_err[1] = mean_rel_err
print(mean_err)