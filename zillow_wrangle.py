## Wrangle File for cleaning data before performing regression analysis ##

#---------------------------Imports----------------------------------------------------------

import env
import pandas as pd
import numpy as np
import os
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

#---------------------------Connection Info Function------------------------------------------

# Connection information from the env file for the mySQL Server

def get_connection(db, user=env.username, host=env.hostname, password=env.password):
    connection_info = f'mysql+pymysql://{user}:{password}@{host}/{db}'
    return connection_info

#---------------------------Data Base Functions-----------------------------------------------

# Function to retrieve the 2017 Zillow Property Data Set from CODEUP's mySQL Server 
def get_zillow_data():
    '''
    Function to retrieve the 2017 Zillow Property Data Set from CODEUP's mySQL Server
    '''
    if os.path.isfile('2017_zillow_properties.csv'):
        df = pd.read_csv('2017_zillow_properties.csv', index_col=0)  # If csv file exists read in data from csv file.
    else:
        sql = '''
                SELECT bedroomcnt, bathroomcnt, 
	            calculatedfinishedsquarefeet, 
	            taxvaluedollarcnt, yearbuilt, 
	            taxamount, fips 
                FROM properties_2017
                WHERE propertylandusetypeid = 261;'''   # SQL query
                                                    
        db = 'zillow'                                   # Database name
        df = pd.read_sql(sql, get_connection(db))       # Pandas DataFrame
        df.to_csv('2017_zillow_hot_month_properties.csv')         # Cache Data
    return df

#---------------------------Function to Clean Zillow Data---------------------------------------

# This function will be used to clean the Zillow data:
def clean_zillow(zillow):
    zillow = zillow.rename(columns = {'bedroomcnt': 'bedrooms',
                                 'bathroomcnt': 'bathrooms',
                                 'calculatedfinishedsquarefeet': 'sqft',
                                 'taxvaluedollarcnt': 'tax_value',
                                 'taxamount': 'tax_amount',
                                 'yearbuilt': 'year_built'})

    zillow = zillow.dropna()    # drop nulls

    # Change bedroom count, year built, calculated finished squarefeet, and fips value type to int
    zillow.bedrooms = zillow.bedrooms.astype('int64')
    zillow.sqft = zillow.sqft.astype('int64')
    zillow.year_built = zillow.year_built.astype('int64')
    zillow.fips = zillow.fips.astype('int64')

    return zillow

#---------------------------Function to Remove Outliers from Zillow Data--------------------------

# Function to remove outliers:
def remove_outliers(df, k=1.5, col_list=[]):
    ''' remove top and bottom 10 % of outliers from a list of 
    columns in a dataframe and return that dataframe'''
    for col in col_list:
        q1, q3 = df[col].quantile([.1, .9])  # get quartiles; Adjusted to remove bottom and top 10%
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#---------------------------Train, Val, Test Function---------------------------------------------

def train_validate_test_split(df, seed=123):
    '''
    This function takes in a dataframe and an integer for a setting a seed
    and splits the data into train, validate and test.  
    '''
    train_and_validate, test = train_test_split(df.drop(columns=['fips']), random_state=seed)
    train, validate = train_test_split(train_and_validate)
    return train, validate, test


#---------------------------Function run all of the Above on Zillow Data (before scaling)----------

def wrangle_zillow():
    '''Function to get zillow data from SQL server, clean it, 
    and then split into train, validate, and test'''
        # Get Data:
    zillow = get_zillow_data()
        # Function to clean the zillow data
    zillow = clean_zillow(zillow)
        # Function to remove the outliers of the zillow data so that they do not affect the regression models
    col_list = ['bedrooms', 'bathrooms', 'sqft', 'tax_value', 'tax_amount']
    zillow = remove_outliers(zillow, 1.5, col_list)
        # Function to split the df into train, validate, and test
    train_and_validate, test = train_test_split(zillow.drop(columns=['fips']), random_state=123)
    train, validate = train_test_split(train_and_validate)

    return train, validate, test


#---------------------------Function to Scale Zillow Data-----------------------------------------

# Function to Scale Zillow Data using min max scaler
def zillow_scaler(train, validate, test):
    '''Min Max Scaler on Train, Validate, Test'''
    columns_to_scale = ['bedrooms', 'bathrooms', 'sqft', 'tax_value', 'tax_amount']

    # 1. create the object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # 2. fit the object (learn the min and max value)
    scaler.fit(train[columns_to_scale])

    # Name Index for scaled columns:
    columns_scaled = ['bedrooms_scaled', 'bathrooms_scaled', 'sqft_scaled', 'tax_value_scaled', 'tax_amount_scaled']
    
    # 3. use the object (use the min, max to do the transformation)
    train[columns_scaled] = scaler.transform(train[columns_to_scale])
    validate[columns_scaled] = scaler.transform(validate[columns_to_scale])
    test[columns_scaled] = scaler.transform(test[columns_to_scale])

    return train, validate, test



def zillow_xy(train, validate, test):
    '''Function to split train, validate, and test
    into their X and y components'''
    # create X & y version of train, where y is a series with just the target variable and X are all the features. 

    X_train = train.drop(columns=['tax_value'])
    y_train = train.tax_value

    X_validate = validate.drop(columns=['tax_value'])
    y_validate = validate.tax_value

    X_test = test.drop(columns=['tax_value'])
    y_test = test.tax_value

    return X_train, y_train, X_validate, y_validate, X_test, y_test