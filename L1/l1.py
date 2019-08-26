# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 22:06:34 2019

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#numOfPoint = 30
#noise = np.random.normal(0,1,numOfPoint).reshape(-1,1)
#x = np.linspace(30, 100, numOfPoint).reshape(-1,1)
#N = x.shape[0]
#y = 15*x + 8 + 20*noise
#plt.scatter(x, y)

data = pd.read_csv('data_linear.csv').values
N = data.shape[0]
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')

x = np.hstack((np.ones((N, 1)), x))

w = np.array([0.,1.]).reshape(-1,1)

numOfIteration = 100
cost = np.zeros((numOfIteration,1))
learning_rate = 0.000001
for i in range(1, numOfIteration):
    r = np.dot(x, w) - y
    cost[i] = 0.5*np.sum(r*r)
    w[0] -= learning_rate*np.sum(r)
    # correct the shape dimension
    w[1] -= learning_rate*np.sum(np.multiply(r, x[:,1].reshape(-1,1)))
    print(cost[i])
predict = np.dot(x, w)
plt.plot((x[0][1], x[N-1][1]),(predict[0], predict[N-1]), 'r')
plt.show()

x1 = 50
y1 = w[0] + w[1] * 50
print('Giá nhà cho 50m^2 là : ', y1)