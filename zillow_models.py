# This file will be used to store the code for the Zillow models
#------------------------------------------------------------------------------------------------------
#---------------Imports--------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns


#------------------------------------------------------------------------------------------------------
#---------------Models---------------------------------------------------------------------------------

def baseline(y_train, y_validate):
    '''Function that will create a baseline prediction'''
    # Turn y into a dataframe
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    # Predict mean:
        #Train
    pred_mean_train = y_train.tax_value.mean()
    y_train['pred_mean'] = pred_mean_train
        #Validate
    pred_mean_validate = y_validate.tax_value.mean()
    y_validate['pred_mean'] = pred_mean_validate
    # Find the Root Mean Squared Error
        #Train
    rmse_y_train = mean_squared_error(y_train.tax_value, y_train.pred_mean) ** .5
        #Validate
    rmse_y_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_mean) ** .5

    print("Baseline RMSE using Mean\nTrain/In-Sample: ", round(rmse_y_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_y_validate, 2))

    return y_train, y_validate


#---------------Linear Regression (OLS)---------------------------------------------------------------

def linear_regression(X_train, y_train, X_validate, y_validate):
    '''Function to perform a linear regression on our data'''
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    # Create
    lm = LinearRegression(normalize=True)
    # Fit
    lm.fit(X_train, y_train.tax_value)
    # Predict
    y_train['pred_lm'] = lm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lm) ** (1/2)

    # predict validate
    y_validate['pred_lm'] = lm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_lm) ** (1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    return rmse_validate, y_validate
    

#---------------LassoLars------------------------------------------------------------------------------

def lasso_lars(X_train, y_train, X_validate, y_validate):
    '''LassoLars regression for Zillow'''
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # create the model object
    lars = LassoLars(alpha=1)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series!
    lars.fit(X_train, y_train.tax_value)

    # predict train
    y_train['pred_lars'] = lars.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lars) ** (1/2)

    # predict validate
    y_validate['pred_lars'] = lars.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_lars) ** (1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    return y_validate


#---------------Tweedie Regressor (GLM)----------------------------------------------------------------

def tweedie_regressor(X_train, y_train, X_validate, y_validate):
    '''Tweedie regressor for zillow data'''
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.tax_value)

    # predict train
    y_train['pred_glm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_glm) ** (1/2)

    # predict validate
    y_validate['pred_glm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_glm) ** (1/2)

    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    return y_validate


#---------------Polynomial Regression------------------------------------------------------------------

def polynomial_regression(X_train, y_train, X_validate, y_validate, degree=2):
    pf = PolynomialFeatures(degree)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    # X_test_degree2 = pf.transform(X_test)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.G3)

    # predict train
    y_train['G3_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.G3, y_train.G3_pred_lm2)**(1/2)

    # predict validate
    y_validate['G3_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.G3, y_validate.G3_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
        "\nValidation/Out-of-Sample: ", rmse_validate)


#------------------------------------------------------------------------------------------------------
#---------------Compare Regression---------------------------------------------------------------------

def make_metric_df(y, y_pred, model_name, metric_df):
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
                'model': model_name,
                'RMSE_validate': round(mean_squared_error(
                    y,
                    y_pred) ** .5),
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name, 
                'RMSE_validate': round(mean_squared_error(
                    y,
                    y_pred) ** .5),
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }, ignore_index=True)

#---------------Compare Regression---------------------------------------------------------------------
def model_compare(X_train, y_train, X_validate, y_validate):
    '''Function to compare regression models'''
    # create the metric_df as a blank dataframe
    metric_df = pd.DataFrame()

    # Baseline
    y_train_b, y_validate_b = baseline(y_train, y_validate)
        # make our first entry into the metric_df with mean baseline
    metric_df = make_metric_df(y_validate_b.tax_value,
                            y_validate_b.pred_mean,
                            'mean_baseline',
                            metric_df)
    # OLS
    rmse_validate, y_validate_ols = linear_regression(X_train, y_train, X_validate, y_validate)

    metric_df = metric_df.append({
    'model': 'OLS Regressor', 
    'RMSE_validate': round(rmse_validate, 0),
    'r^2_validate': explained_variance_score(y_validate_ols.tax_value, y_validate_ols.pred_lm)}, ignore_index=True)

    # LassoLars
    y_validate_ll = lasso_lars(X_train, y_train, X_validate, y_validate)
    metric_df = make_metric_df(y_validate_ll.tax_value,
               y_validate_ll.pred_lars,
               'lasso_alpha_1',
               metric_df)

    # Tweedie regressor
    y_validate_t = tweedie_regressor(X_train, y_train, X_validate, y_validate)
    metric_df = make_metric_df(y_validate_t.tax_value,
                y_validate_t.pred_glm,
                'glm_poisson',
                metric_df)

    return metric_df


#------------------------------------------------------------------------------------------------------
#---------------plot Regression------------------------------------------------------------------------
def plot_baseline(y_train, y_validate):
    y_train, y_validate = baseline(y_train, y_validate)
    # plot to visualize actual vs predicted. 
    plt.hist(y_train.tax_value, color='blue', alpha=.5, label="Samples")
    plt.hist(y_train.pred_mean, bins=1, color='red', alpha=.5, rwidth=1000, label="Mean")
    #plt.xlabel("Home Value")
    #plt.ylabel("not sure")
    plt.legend()
    plt.show()
    return None

