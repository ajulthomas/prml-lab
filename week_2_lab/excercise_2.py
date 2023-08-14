# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 19:01:52 2023

@author: ajul_thomas
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([[1200, 2, 1, 1995],
              [1500, 3, 2, 2002],
              [1800, 3, 2, 1985],
              [1350, 2, 1, 1998],
              [2000, 4, 3, 2010]])

y = np.array([250, 320, 280, 300, 450])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict house prices
new_data = np.array([[1650, 3, 2, 2005],
                     [1400, 2, 1, 2000]])

predicted_prices = model.predict(new_data)
print("Predicted prices:", predicted_prices)


