#####================= Import Libraries =================
# Importing the required packages
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
import scipy
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

####================Import and read the dataset=================


# Read florence airbnb dataset as panda dataframe
df_orig = pd.read_csv("listings.csv",parse_dates=['host_since','first_review','last_review'])
df_orig = pd.DataFrame(df_orig)


# Setting 'id'column as the index column and display first few results of airbnb subset dataframe that we will follow further
df_orig = df_orig.set_index('id')            

#display the first three results again for our reference
df_orig.head(3)

###===============Explore data: getting columns of dataset for overview =================
print(df_orig.columns.values)
# we can also get the list view of columns from below command
#list(df_orig.columns)

#shape of the dataset
print(df_orig.shape)

##===========Copying the Original Dataset to new dataframe name 'df_airbnb' ===================


df_airbnb = df_orig.copy()

######### Dropping some columns because of non relevant information.
# For example there are urls, host id,country name, and Full Text Attributes which are of no use.
# These columns will all be attributes that have natural language, and we don’t have a need for them. So we drop them (pd.drop)



col_drop = df_airbnb.loc[:, ['listing_url', 'scrape_id','last_scraped','name','summary', 'space','description','experiences_offered', 
            'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules','thumbnail_url', 
            'medium_url', 'picture_url','xl_picture_url','host_url','host_id','host_name','host_location','host_about', 
            'host_thumbnail_url','host_picture_url','host_neighbourhood','host_total_listings_count', 'host_verifications',
            'host_has_profile_pic','host_identity_verified','street', 'neighbourhood_cleansed','neighbourhood_group_cleansed',
            'city','state', 'zipcode','market','smart_location','country_code','country','is_location_exact','weekly_price',
            'monthly_price','maximum_nights','minimum_minimum_nights','maximum_minimum_nights','minimum_maximum_nights',
            'maximum_maximum_nights','minimum_nights_avg_ntm','maximum_nights_avg_ntm','calendar_updated',
            'has_availability','calendar_last_scraped','review_scores_accuracy','review_scores_cleanliness',
            'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value',
            'requires_license','license','jurisdiction_names','is_business_travel_ready','require_guest_profile_picture',
            'require_guest_phone_verification','calculated_host_listings_count','calculated_host_listings_count_entire_homes',
            'calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms',]]


df_airbnb = df_airbnb.drop(col_drop, axis =1)

print(df_airbnb.shape)
print(df_airbnb.head(3))

# #### =================DATASET INFO & DESCRIPTION =================


print("Datatype of Features about Features:")
print(df_airbnb.dtypes)
print ('='*80 + '\n')
print ('='*80 + '\n')
print("Description about Features:")
print(df_airbnb.describe())
print ('='*80 + '\n')
print ('='*80 + '\n')
print("Brief Information about Dataset:")
print(df_airbnb.info())
print ('='*80 + '\n')
print ('='*80 + '\n')


## ================= INITIAL PROCESSING =================
# 
# Random Forest: A machine learning technique for predicting category/continous variable. Can predict with data of any type.
# It doesn't generally overfits. Its a great place to start with.
# Here if we try to run the model to fit on to data then it will not work because dataset is not
# cleaned and has missing values that we have to deal with in our further work.

# from sklearn.ensemble import RandomForestRegressor
# X  = df_airbnb.drop('price', axis =1)
# y = df_airbnb.price
# rfg = RandomForestRegressor()
# rfg.fit(X,y)

## If we try to run above code to fit model then it will generate the error like below:
# TypeError: float() argument must be a string or a number, not 'Timestamp'


## ================= Handling and Visualizing Missing values in dataset ======================


print("Missing value and dtypes of the data")
missingval_dtypes = pd.concat([df_airbnb.isnull().sum(), df_airbnb.dtypes], axis =1 )
# print(missingval_dtypes.rename(columns={0:'nans',1:'dtypes'}))
print(missingval_dtypes)

# ------- Plotting missing value graph for visualization of our daaset ---------
# calculating percentage missing values for each feature using bar graph in matplotlib
missing_values = df_airbnb.isnull().sum() / df_airbnb.shape[0]*100
pmv = missing_values.plot(kind = 'bar', color='blue', figsize = (20, 10),)
pmv.set_xlabel('Features', fontsize=15)
pmv.set_ylabel('Missing Data', fontsize=15)
pmv.set_title('Percentage of Missing values', fontsize=20)
plt.show()

print('From the above Graph it is clear that some of the columns are with more than 30% missing '
      'values then its wise to drop them right away from the dataset as they are of no use.')



#Here we are dropping columns with more than 30% missing values.
missing_30perc_val = df_airbnb.columns[missing_values > 30]  #taking percentage of_missing_values from above

df_airbnb = df_airbnb.drop(missing_30perc_val, axis=1);


# updated dataset after dropping columns with more than 30% missing values can be checked with below code again for reference
# print(df_airbnb.isnull().sum())                                        


# Visualizing Dataset after dropping features with more than 30% missing values
# BIFURCATE DATASET INTO 3 SUBSETS FROM ORIGINAL DATASET
df_plot1 = df_airbnb.iloc[:, 0:12]
df_plot2 = df_airbnb.iloc[:, 12:24]
df_plot3 = df_airbnb.iloc[:, 24:37]

#SHAPE OF THE THREE SUBSETS
print(df_plot1.shape, df_plot2.shape, df_plot3.shape)

#PLOTTING GRAPHS FOR THREE SUBSETS OF ORIGINAL DATASET
# we have divided dataset in to three for our ease so as to visualize all the columns clearly for analysing missing values
plot1 = df_plot1.isnull().sum() / df_orig.shape[0]*100
pl1 = plot1.plot(kind = 'bar', color='green', figsize = (20, 10),)
pl1.set_xlabel('Features', fontsize=20)
pl1.set_ylabel('Missing Data', fontsize=15)
pl1.set_title('PLOT1 - Percentage of Missing Values ', fontsize=18)

plt.show()
print('#',50*"-")

plot2 = df_plot2.isnull().sum() / df_orig.shape[0]*100
pl2 = plot2.plot(kind = 'bar', color='orange', figsize = (20, 10),)
pl2.set_xlabel('Features', fontsize=20)
pl2.set_ylabel('Missing Data', fontsize=20)
pl2.set_title('PLOT2 - Percentage of Missing Values', fontsize=20)

plt.show()

plot3 = df_plot3.isnull().sum() / df_orig.shape[0]*100
pl3 = plot3.plot(kind = 'bar', color='red', figsize = (20, 10),)
pl3.set_xlabel('Features', fontsize=20)
pl3.set_ylabel('Missing Data', fontsize=15)
pl3.set_title('PLOT3 - Percentage of Missing Values', fontsize=18)

plt.show()




#DESCRIPTION AND SHAPE OF THE NEW DATAFRAME "df_airbnb" FOR REFERENCE again
print(df_airbnb.describe())
print(df_airbnb.shape)



# CHECKING THE NANS AND DTYPES SIMULATNEOUSLY IN TABLE FORMAT OF DATAFRAME "df_airbnb"
print("Initially checking Misiing value and dtypes of the data in table form")
missingval_dtypes = pd.concat([df_airbnb.isnull().sum(), df_airbnb.dtypes], axis =1 )
dataset_nans_dtypes = missingval_dtypes.rename(columns={0:'nans',1:'dtypes'})
print(dataset_nans_dtypes)


### ==================== Data Manipulation for Price Columns: Price from string to integer ===============



# Transform pricing features into integer.
#Remove '$' and ',' sign from 'price', 'security_deposit' and 'security_deposit' columns and convert to integer


df_airbnb['price']= df_airbnb['price'].str.replace('$', '').str.replace(',', '')
df_airbnb['price'] = pd.to_numeric(df_airbnb['price'],errors = 'coerce')

df_airbnb['security_deposit']= df_airbnb['security_deposit'].str.replace('$', '').str.replace(',', '') 
df_airbnb['security_deposit'] = pd.to_numeric(df_airbnb['security_deposit'],errors = 'coerce')

df_airbnb['cleaning_fee']= df_airbnb['cleaning_fee'].str.replace('$', '').str.replace(',', '')
df_airbnb['cleaning_fee'] = pd.to_numeric(df_airbnb['cleaning_fee'],errors = 'coerce')

df_airbnb['extra_people']= df_airbnb['extra_people'].str.replace('$', '').str.replace(',', '')
df_airbnb['extra_people'] = pd.to_numeric(df_airbnb['extra_people'],errors = 'coerce')



# --------------Getting the CSV FILE FOR CLEANED PRICE COLUMN TO BE USED IN GUI -----------
for col in df_airbnb.columns[df_airbnb.isnull().any()]:
    print(col)

airbnb_price = df_airbnb.to_csv("airbnb_price.csv", index=None)
print("csv created")
# ----------------------------------------------------------------------------------------
# We here reviewed the datatype again after conversion if it has been implemented successfully
#print(df_orig[['price','security_deposit','cleaning_fee']].dtypes)

# Description of the TARGET feature
df_airbnb.describe()["price"]

print("Description of target: price suggests that there might be outliers - 75% properties have a price that is lower than 110 yet the highest " \
                                       "price is 6000. We need to further investigate the outliers and remove them if necessary")




## -----------Boxplot and Distribution plot of price column to check outliers ----------
boxplot_price = dict(markerfacecolor='r', markeredgecolor='r', marker='.')
df_airbnb['price'].plot(kind='box', xlim=(0, 1000), vert=False, figsize=(16,2));
plt.show()

# -----Distribution Plot for column 'Price'------
print ('='*80 + '\n')

print("Mean of the price column:", df_airbnb['price'].mean())
print("Median value of the price column:", df_airbnb['price'].median())
print ('='*80 + '\n')


# also plotting  median and mean verticle line to check the deviation
plt.hist(df_airbnb['price'],color='c', bins=25);
plt.axvline(x=df_airbnb['price'].mean(),linestyle='dashed', linewidth=1, color='r');
plt.axvline(x=df_airbnb['price'].median(),linestyle='dashed', linewidth=1,color='b' );
plt.xlabel('price distribution from the mean and median')
plt.show()

print("The boxplot for price shows quite a skewed distribution with a long tail of high-priced outliers. "
      "However, 75% of all airbnbs only cost up to 110 Dollar. For this project, we can remove the extremely high priced"
      " rentals above Dollar 380/night to maintain comparability")
print ('='*80 + '\n')


#Removing outliers from the target "Price" to avoid wrong results and assumptions
df_airbnb.drop(df_airbnb[df_airbnb['price'] > 380].index, axis=0, inplace=True)
df_airbnb['price'].describe()



#lets check description of cleaning fee and security group
print(df_airbnb.loc[:, "cleaning_fee"].describe())
print(df_airbnb.loc[:, "security_deposit"].describe())


# # boxplot of security_deposit column to view the description and outliers
boxplot_cleaning_fee = dict(markerfacecolor='r', markeredgecolor='r', marker='.')
df_airbnb['cleaning_fee'].plot(kind='box', xlim=(0, 1000), vert=False, figsize=(18,3));
plt.show()

# # boxplot of security_deposit column to view the description and outliers
boxplot_security_deposit = dict(markerfacecolor='r', markeredgecolor='r', marker='.')
df_airbnb['security_deposit'].plot(kind='box', xlim=(0, 1000), vert=False, figsize=(18,3));
plt.show()

print('The box plot distribution and description of the security fee and cleaning fee shows that 75% of all airbnbs has only cleaning upto '
      '50 Dollar and security deposit upto 150 Dollar (with 50% of security fee is zero as per distribution.' )


## =============== Data Imputation for cleaning fee and security deposit ================================



# we are doing the median imputation for cleaning_fee and security_deposit
df_airbnb.loc[:,'cleaning_fee'].fillna(df_airbnb['cleaning_fee'].median(), inplace=True)
df_airbnb.loc[:,'security_deposit'].fillna(df_airbnb['security_deposit'].median(), inplace=True)

# Lets check the null/missing values again to further work
df_airbnb.isnull().sum()


## ========= Handling DateTime Columns============
# We have three nice Date Attributes which we can use date_attributes = ['host_since', 'first_review', 'last_review']
# Here, We will extract Year and month from DateTime format 



#Extracting year out of the datetime format of the features
# we have already parsed these columns in datetime format in the beginning for our ease
# we are just using year out of it

df_airbnb['host_sinceyear'] = df_airbnb.loc[:,'host_since'].dt.year

df_airbnb['first_reviewedyear'] = df_airbnb.loc[:, 'first_review'].dt.year

df_airbnb['last_reviewedyear'] = df_airbnb.loc[:,'last_review'].dt.year

# Dropping main date column as we don't need them as redundant
df_airbnb.drop(['host_since'], axis=1, inplace=True)
df_airbnb.drop(['first_review'], axis=1, inplace=True)
df_airbnb.drop(['last_review'], axis=1, inplace=True)

print ('='*80 + '\n')
print(df_airbnb.head())
print ('='*80 + '\n')

## ========== Handling some more features  ============

# 1) host_response_rate is in percentage so we convert it to float ranging between 0.0 < f < 1.0.
# 2) host_is_superhost and instant_bookable converted from boolean to float/integer
# 3) amenities feature is  expanded into dummy variables (considering only top 10 amenities here)


# coverting response rate to float values for range between 0 to 1 for easier analysis
df_airbnb['host_response_rate'] = df_airbnb['host_response_rate'].str.replace('\%','').astype(float)/100



# changing 'host_is_superhost' categorical feature to float and plot the count distribution
set(df_airbnb.loc[:,'host_is_superhost'])
df_airbnb.loc[:,'host_is_superhost'] = df_airbnb.loc[:,'host_is_superhost'].map({'f':0,'t':1}).astype(bool).astype(float)
sns.countplot(x='host_is_superhost',data=df_airbnb)

plt.show()



# changing 'instant_bookable' categorical feature to float and plot the count distribution
set(df_airbnb.loc[:,'instant_bookable'])
df_airbnb.loc[:,'instant_bookable'] = df_airbnb.loc[:,'instant_bookable'].map({'f':0,'t':1}).astype(bool).astype(float)
sns.countplot(x='instant_bookable',data=df_airbnb)

plt.show()

# ====== Extracting 'Ameneties' features  ======

df_airbnb.loc[:,'amenities'].head(3)


# Analyzing another feature 'ameneties'
# This module implements specialized container datatypes providing alternatives to Python’s general
# purpose built-in containers, dict, list, set, and tuple.
# A Counter is a container that keeps track of how many times equivalent values are added.
from collections import Counter
results = Counter()
df_airbnb['amenities'].str.strip('{}').str.replace('"', '') .str.lstrip('\"').str.rstrip('\"').str.split(',').apply(results.update)
results.most_common()


# create a new sub dataframe with 'amenity' and 'count'
sub_amenities = pd.DataFrame(results.most_common(10), columns=['amenity', 'count'])

# ploting the top 10 amenities 
sub_amenities.sort_values(by=['count'], ascending=True).plot(kind='barh', x='amenity', y='count',  
                                                      figsize=(15,15), legend=False, color='green',
                                                      title='Feature_Amenities')
plt.xlabel('Count');




# Considering only top 10 amenities and adding these new ameneties features in the dataset for analysis
df_airbnb['amenities_ Wifi'] = df_airbnb['amenities'].str.contains('Wifi').astype(bool).astype(float)
df_airbnb['amenities_Iron'] = df_airbnb['amenities'].str.contains('Iron').astype(bool).astype(float)
df_airbnb['amenities_Air conditioning'] = df_airbnb['amenities'].str.contains('Air conditioning').astype(bool).astype(float)
df_airbnb['amenities_Heating'] = df_airbnb['amenities'].str.contains('Heating').astype(bool).astype(float)
df_airbnb['amenities_Hair dryer'] = df_airbnb['amenities'].str.contains('Hair dryer').astype(bool).astype(float)
df_airbnb['amenities_Essentials'] = df_airbnb['amenities'].str.contains('Essentials').astype(bool).astype(float)
df_airbnb['amenities_Kitchen'] = df_airbnb['amenities'].str.contains('Kitchen').astype(bool).astype(float)
df_airbnb['amenities_Hangers'] = df_airbnb['amenities'].str.contains('Hangers').astype(bool).astype(float)
df_airbnb['amenities_TV'] = df_airbnb['amenities'].str.contains('TV').astype(bool).astype(float)
df_airbnb['amenities_Laptop friendly workspace'] = df_airbnb['amenities'].str.contains('Laptop friendly workspace').astype(bool).astype(float)

# after adding new features drop original redundant 'amenities' column
df_airbnb.drop(['amenities'], axis=1, inplace=True)

# Check the existing columns on the dataset
print ('='*80 + '\n')
print(df_airbnb.head())
print ('='*80 + '\n')



# ================= Final Cleanup: Dropping all the missing values ==============
df_airbnb = df_airbnb.dropna()
print(df_airbnb.shape)


print("Misiing value and dtypes of the data in table for last check up of dataset")
result = pd.concat([df_airbnb.isnull().sum(), df_airbnb.dtypes], axis =1 )
result1 = result.rename(columns={0:'nans',1:'dtypes'})
print('result1')




# Creating csv file for the cleaned data
for col in df_airbnb.columns[df_airbnb.isnull().any()]:
    print(col)
    
df_airbnb.to_csv("airbnb_cleaned.csv", index=None)
print("csv created")
print(df_airbnb.shape)


## ===== Select non-numeric variables and create dummies (Hot encode nominal Categorical features) =======



#fetching categorical variables in one liner code
categorical_vars = df_airbnb.select_dtypes(include=['object']).columns
df_airbnb[categorical_vars].head()




#Encoding and display Dummy Variable columns
dummy_categorical_vars = pd.get_dummies(df_airbnb[categorical_vars])
print(dummy_categorical_vars.head())



#Drop categorical variables from df_airbnb and add the dummies
df_airbnb=df_airbnb.drop(categorical_vars,axis=1)
airbnb_predictors = pd.merge(df_airbnb,dummy_categorical_vars, left_index=True, right_index=True)
print(airbnb_predictors.head())


# Now the shape must have been different than before as we have added many dummy variables
print(airbnb_predictors.shape)


# ========= CORRELATION MATRIX PLOT ==============
# Correlation matrix is the first thing that we will do to check the best features out of all for modeling and to check multicollinearity


corr = airbnb_predictors.corr()

fig = plt.figure(figsize=(50,50))
sns.heatmap(
    corr,
    vmin=-1,
    vmax=1,
    center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True);
plt.show()
print("Its really hard to analyse so many features at once so we will try to find the best features out of "
      "whole dataset using dimensionality reduction in our further work ")



## ====================== Creating csv for airbnb features to be used in features selection===========================



for col in airbnb_predictors.columns[airbnb_predictors.isnull().any()]:
    print(col)

airbnb_predictors.to_csv("airbnb_features.csv", index=None)
print(" updated csv created")
print(airbnb_predictors.shape)
