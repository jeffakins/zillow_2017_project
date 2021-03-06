## Wrangle File for cleaning data before performing regression analysis ##

#---------------------------Imports----------------------------------------------------------

import env
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

#---------------------------------------------------------------------------------------------
#---------------------------Connection Info Function------------------------------------------

# Connection information from the env file for the mySQL Server

def get_connection(db, user=env.username, host=env.hostname, password=env.password):
    connection_info = f'mysql+pymysql://{user}:{password}@{host}/{db}'
    return connection_info

#---------------------------------------------------------------------------------------------
#---------------------------Data Base Function------------------------------------------------

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
                    taxamount, fips, regionidzip 
                FROM properties_2017
                JOIN predictions_2017 USING(id)
                WHERE propertylandusetypeid = 261
                    AND transactiondate BETWEEN '2017-05-01' AND '2017-09-01';'''       
                # SQL query
                                                    
        db = 'zillow'                                       # Database name
        df = pd.read_sql(sql, get_connection(db))           # Pandas DataFrame
        df.to_csv('2017_zillow_hot_month_properties.csv')   # Cache Data
    return df


#---------------------------Function to Clean Zillow Data---------------------------------------

# This function will be used to clean the Zillow data:
def clean_zillow(zillow):
    zillow = zillow.rename(columns = {'bedroomcnt': 'bedrooms',
                                 'bathroomcnt': 'bathrooms',
                                 'calculatedfinishedsquarefeet': 'sqft',
                                 'taxvaluedollarcnt': 'tax_value',
                                 'taxamount': 'tax_amount',
                                 'yearbuilt': 'year_built',
                                 'regionidzip': 'zipcode'})

    zillow = zillow.replace(r'^\s*$', np.nan, regex=True) # Format nulls
    zillow = zillow.dropna()    # drop nulls

    return zillow

#---------------------------Function to Remove Outliers from Zillow Data----------------------

# Function to remove outliers:
def remove_outliers(df, k=1.5, col_list=[]):
    ''' remove top and bottom 10 % of outliers from a list of 
    columns in a dataframe and return that dataframe'''
    for col in col_list:
        q1, q3 = df[col].quantile([.2, .8])  # get quartiles; Adjusted to remove bottom and top 10%
        iqr = q3 - q1   # calculate interquartile range
            
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df

#---------------------------------------------------------------------------------------------
#---------------------------Zipcode Function--------------------------------------------------

def get_zipcode_data():
    '''
    Function to retrieve the 2016 Zillow average home price per zipcode data from CODEUP's mySQL Server
    '''
    if os.path.isfile('2016_zillow_zipcodes.csv'):
        df = pd.read_csv('2016_zillow_zipcodes.csv', index_col=0)  # If csv file exists read in data from csv file.
    else:
        sql = '''
                SELECT COUNT(regionidzip) AS zipcode_count, 
                    regionidzip AS zipcode, 
                    ROUND(AVG(taxvaluedollarcnt),0) AS zipcode_avg_price
                FROM properties_2016
                WHERE propertylandusetypeid = 261
                GROUP BY regionidzip
                ORDER BY AVG(taxvaluedollarcnt) DESC;'''   # SQL query
                                                    
        db = 'zillow'                                   # Database name
        df = pd.read_sql(sql, get_connection(db))       # Pandas DataFrame
        df.to_csv('2016_zillow_zipcodes.csv')         # Cache Data
    return df

#---------------------------Function to clean Zipcode Data---------------------------------------

def clean_zipcode(zips):
    '''Function removes nulls from zipcode data and converts all numbers to int'''
    zips = get_zipcode_data()
    zips = zips.replace(r'^\s*$', np.nan, regex=True)
    zips = zips.dropna()
    zips = zips.astype('int64')
    return zips

#---------------------------------------------------------------------------------------------
#-----------------Function run all of the Above on Zillow Data (before scaling)---------------

def wrangle_zillow():
    '''Function to get zillow data from SQL server, clean it, 
    and then combine it will 2016 average home price based on zipcode location'''
        # Get Zillow Data:
    zillow = get_zillow_data()
        # Function to clean the zillow data:
    zillow = clean_zillow(zillow)

    # Get Zipcode Data:
    zips = get_zipcode_data()
    # Clean zipcode data:
    zips = clean_zipcode(zips)

    # Joining 2016 average home price by zipcode to the zillow df:
    zillow = zillow.merge(zips, left_on='zipcode', right_on='zipcode', how='outer', indicator=True)
    # Removing nulls
    zillow = zillow.dropna()

    # Change bedroom count, year built, calculated finished squarefeet, and fips value type to int
    zillow.bedrooms = zillow.bedrooms.astype('int64')
    zillow.sqft = zillow.sqft.astype('int64')
    zillow.year_built = zillow.year_built.astype('int64')
    zillow.fips = zillow.fips.astype('int64')
    zillow.zipcode = zillow.zipcode.astype('int64')
    # Drop uneeded columns after merge:
    zillow = zillow.drop(columns=['zipcode_count', '_merge'])

    # Function to remove the outliers of the zillow data so that they do not affect the regression models:
    col_list = ['bedrooms', 'bathrooms', 'sqft', 'tax_value', 'zipcode_avg_price']
    zillow = remove_outliers(zillow, 1.5, col_list)

    # fips lacation names:
    fips = {'fips': [6037, 6059, 6111],
        'county': ['Los Angeles County', 'Orange County', 'Ventura County'],
        'state': ['CA', 'CA', 'CA']}
    # fips dataframe:
    fips_name = pd.DataFrame(data=fips)
    # Merge zillow df with fips df: 
    zillow = zillow.merge(fips_name, left_on='fips', right_on='fips', how='outer', indicator=False)

    return zillow


#---------------------------------------------------------------------------------------------
#---------------------------Function to Scale Zillow Data-------------------------------------

# Function to Scale Zillow Data using min max scaler
def zillow_scaler(train, validate, test):
    '''Min Max Scaler on Train, Validate, Test'''
    columns_to_scale = ['bedrooms', 'bathrooms', 'sqft', 'zipcode_avg_price']

    # 1. create the object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # 2. fit the object (learn the min and max value)
    scaler.fit(train[columns_to_scale])

    # Name Index for scaled columns:
    # columns_scaled = ['bedrooms_scaled', 'bathrooms_scaled', 'sqft_scaled', 'tax_value_scaled', 'tax_amount_scaled']
    
    # 3. use the object (use the min, max to do the transformation)
    train[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test[columns_to_scale] = scaler.transform(test[columns_to_scale])

    return train, validate, test

#---------------------------------------------------------------------------------------------
#---------------------------Train, Val, Test Function-----------------------------------------

def train_validate_test_split(df, seed=123):
    '''
    This function takes in a dataframe and an integer for a setting a seed
    and splits the data into train, validate and test.  
    '''
    train_and_validate, test = train_test_split(df, random_state=seed)
    train, validate = train_test_split(train_and_validate)
    return train, validate, test

#---------------------------------------------------------------------------------------------
#---------------------------Function to split T, V, T into X, y-------------------------------

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



#---------------------------------------------------------------------------------------------
#---------------------------Plotting----------------------------------------------------------
# Scatterplot:
def initial_plot(df, x, y):
    '''Plots initial scatterplot with a trend line'''
    plt.figure(figsize = (9,6), facecolor='aliceblue')
    sns.set_theme(style='whitegrid')
    sns.color_palette('tab10')

    #graph = sns.lmplot(x=x, y=y, data=df, scatter=True, hue=None, col=None)
    sns.lmplot(x=x, y=y, data=df, scatter=True, hue=None, col=None)

    #ylabels = [(lambda x, pos: '{:.1f} m'.format(x / 1_000_000)]
    #graph.set_ylabels(ylabels)
    plt.title(f'Plot of No. {x} vs {y} with Regression Line', fontsize = 12, pad=20)
    plt.show()
    return None

# Subplots:
def zillow_subplots(zillow):
    '''Plots histogram subplots'''
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in zillow.columns]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        zillow[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)
        
        plt.tight_layout()
        
        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)
    return None