# This file will be used to store the code for the Zillow models

#---------------Imports----------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

#---------------Baseline---------------------------------------------------------------------

def baseline(y):
    '''Function that will create a baseline prediction'''
    # Turn y into a dataframe
    y = pd.DataFrame(y)

    # Predict mean
    pred_mean = y.iloc[:,1].mean()
    y['pred_mean'] = pred_mean
    
    # Find the Root Mean Squared Error
    rmse_y = mean_squared_error(y.iloc[:,1], y.pred_mean) ** .5

    print("RMSE using mean: ", round(rmse_y, 2))

    return y


#---------------Linear Regression (OLS)------------------------------------------------------




#---------------LassoLars--------------------------------------------------------------------




#---------------Tweedie Regressor (GLM)------------------------------------------------------




#---------------Polynomial Regression--------------------------------------------------------