import tensorflow as tf
import tensorflow_probability as tfp
import time

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tfd = tfp.distributions
ode = tfp.math.ode

# 定义洛伦兹方程
def lorenz_eqn(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = tf.unstack(state)
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return tf.stack([dxdt, dydt, dzdt])

# 初始状态
state0 = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)

# 时间范围
t = tf.linspace(0.0, 25.0, 1000)

# 定义 ODE solver
solver = ode.DormandPrince()

# 使用ODE solver求解洛伦兹方程
start = time.time()
results = ode(lambda t, y: lorenz_eqn(t, y), state0, t, atol=1e-6, rtol=1e-6, method=solver)
time.sleep(2)
end = time.time()

print(start-end)