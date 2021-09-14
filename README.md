![TELCO Logo](https://lh3.googleusercontent.com/proxy/w0hjSS3KvVmQpSSXUnfGDiN-TsOuK6qwDe29cfb6eRgBP-ULpqjCiTtmSsueUITHdLeIlegGYTg5-_ateni9dwzz6CcO--2cul58-IIOCZ4FMMbh7Ht-g4V5xgywqcxlKm1LzBVmKk8Bozqe)

# Zillow Regression Project
### - by Jeff Akins

## Project Summary
### Business Goals
- Predict the values of single unit properties that the tax district assesses using the property data from those with a transaction during the "hot months" (in terms of real estate demand) of May-August, 2017
#### Additional Goals
- Property taxes are assessed at the county level; therefore, we would like to know what states and counties these are located in.
- Clearly show the distribution of tax rates for each county.

## Executive Summary
- 
- 

## Deliverables
- A report in the form of a presentation, verbal supported by slides.
- The report/presentation slides should summarize your findings about the drivers of the single unit property values. This will come from the analysis you do during the exploration phase of the pipeline. In the report, you should have visualizations that support your main points.
- A github repository containing your work.
 - Clearly labeled final Jupyter Notebook that walks through the pipeline. 
 - Evaluate your model by computing the model metrics and comparing against a baseline.
 - Any .py files necessary to reproduce your work.
 - This README.md file.

### Data dictionary
Index | Column Name | Description | Count | Dtype
--|--|--|--
 
 0 |  bedrooms          | Number of Bedrooms                                 | 29789 non-null | int64  
 1 |  bathrooms         | Number of bathrooms                                | 29789 non-null | float64
 2 |  sqft              | Square footage of the house                        | 29789 non-null | int64  
 3 |  tax_value         | Value of the property                              | 29789 non-null | float64
 4 |  year_built        | Year property was built                            | 29789 non-null | int64  
 5 |  tax_amount        | Tax amount per property                            | 29789 non-null | float64
 6 |  fips              | Federal Information Processing Series (FIPS) Codes | 29789 non-null | int64  
 7 |  zipcode           | Zipcode                                            | 29789 non-null | int64  
 8 |  zipcode_avg_price | Average home price per zipcode                     | 29789 non-null | int64 

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

### Prep

### Explore

### Model & Evaluate

## Conclusion