![TELCO Logo](http://www.sustainablerealtygroup.com/wp-content/uploads/2016/04/Zestimate-Image.jpg)

# Zillow Regression Project
### - by Jeff Akins

## Project Summary
### Business Goals
- Predict the values of single unit properties that the tax district assesses using the property data from those with a transaction during the "hot months" (in terms of real estate demand) of May-August, 2017
#### Additional Goals
- Property taxes are assessed at the county level; therefore, we would like to know what states and counties these are located in.
- Clearly show the distribution of tax rates for each county.

## Executive Summary
- After modeling the zillow data using four features (bedrooms, bathrooms, square footage, and 2016 average home price by zip code), the OLS and Lasso + Lars produced the best results with a 19% improvement over the baseline model (when comparing validation data). Therefore, these models do show that they could be used to produce a prediction for home values; however, the error is still high at over $180,000 RSME. Additional refining would need to be done in order to use this model as a reliable predictor of home values. 


## Deliverables
- A report in the form of a presentation, verbal supported by slides.
- The report/presentation slides should summarize your findings about the drivers of the single unit property values. This will come from the analysis you do during the exploration phase of the pipeline. In the report, you should have visualizations that support your main points.
- A github repository containing your work.
 - Clearly labeled final Jupyter Notebook that walks through the pipeline. 
 - Evaluate your model by computing the model metrics and comparing against a baseline.
 - Any .py files necessary to reproduce your work.
 - This README.md file.

### Data dictionary

|Index | Column Name | Description | Count | Dtype|
|---|---|---|---|---|
|0 |  bedrooms          | Number of Bedrooms                                 | 27363 non-null | int64  |
|1 |  bathrooms         | Number of bathrooms                                | 27363 non-null | float64|
|2 |  sqft              | Square footage of the house                        | 27363 non-null | int64  |
|3 |  tax_value         | Value of the property                              | 27363 non-null | float64|
|4 |  year_built        | Year property was built                            | 27363 non-null | int64  |
|5 |  tax_amount        | Tax amount per property                            | 27363 non-null | float64|
|6 |  fips              | Federal Information Processing Series (FIPS) Codes | 27363 non-null | int64  |
|7 |  zipcode           | Zipcode                                            | 27363 non-null | int64  |
|8 |  zipcode_avg_price | Average home price per zipcode                     | 27363 non-null | int64  |
|9 |  county            | County Name                                        | 27363 non-null | object |
|10|  state             | State Name                                         | 27363 non-null | object |

## Project Specifications

### Plan:
- Determine Data needed
- Bring in appropriate data from Zillow database in SQL (Acquire)
- Prep
- Explore
- Model
- Iterate - meaning attempt to make the model better
- Deliver 

### Acquire
- I created a series of functions to acquire and clean the zillow data, which are located in the zillow_wrangle.py file. They take in all of the single unit properties (code 261 from the propertylandusetypeid column on the properties_2017 table in the zillow data set) that were sold between 1 May and 1 Sept of 2017 (based on the predictions_2017 table) into a Pandas dataFrame.
- I then add on a column for zipcodes and average home price per zipcode, as well as the county and sate name based on the FIPS code.

### Prep
- Top and Bottom 20% of outliers were removed.
- Uneeded columns were removed
- The remaining data, aside from the target, was scaled using a min-max scaler

### Explore
- Scatterplots with trend lines are used to look for obvious coorelations
- Boxplots are used to show the distributions
- A heatmap was generated to show correlation strength between features
- Statistical tests were used to confirm correlation between the target and other features

### Model & Evaluate
- The following models were used:
 - Baseline (using mean)
 - Ordinary Least Squares
 - LASSO + LARS (alpha = 1)
 - Generalized Linear Model (power = 1: Poisson Distribution)

## Conclusion
- The OLS and Lasso + Lars produced the best results with a 19% improvement over the baseline model (comparing validation data). Therefore, these models do show that they could be used to produce a prediction for home values; however, the error is still high at over $180,000 RSME. Additional refining would need to be done in order to use this model as a reliable predictor of home values.