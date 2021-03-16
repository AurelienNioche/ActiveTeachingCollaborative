import probflow as pf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

randn = lambda *x: np.random.randn(*x).astype('float32')

# Generate some data
x = randn(100)
y = 2*x -1 + randn(100)
x.reshape(-1, 1)
y.reshape(-1, 1)

# Plot it
# plt.plot(x, y, '.')


pf.set_backend('pytorch')

class SimpleLinearRegression(pf.ContinuousModel):

    def __init__(self):
        self.w = pf.Parameter(name='Weight')
        self.b = pf.Parameter(name='Bias')
        self.s = pf.ScaleParameter(name='Std')

    def __call__(self, x):
        x = torch.tensor(x)
        return pf.Normal(x*self.w()+self.b(), self.s())

# model = pf.LinearRegression(1)
model = SimpleLinearRegression()
model.fit(x, y)
model.posterior_plot(ci=0.95)
plt.show()