# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:56:14 2019

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hàm sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Toán tử AND
plt.scatter([1], [1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter([0, 0, 1], [0, 1, 0], c='blue', edgecolors='none', s=30, label='từ chối')
plt.plot([0, 1.5], [1.5, 0], 'g')

# Toán tử OR
plt.scatter([0, 1, 1], [1, 0, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter([0], [0], c='blue', edgecolors='none', s=30, label='từ chối')
plt.plot([-0.5, 1.5], [1, -1], 'g')
plt.xlabel('x1')
plt.ylabel('x2')

# Toán tử XOR
plt.scatter([1, 0], [0, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter([1, 0], [1, 0], c='blue', edgecolors='none', s=30, label='từ chối')
plt.xlabel('x1')
plt.ylabel('x2')
