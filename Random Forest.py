
# ===================== RANDOM FOREST FOR FEATURE SELECTION ==================

# Importing the required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


#%%-----------------------------------------------------------------------
# In[1]:
# import Dataset
data = pd.read_csv('airbnb_features.csv')

# In[2]:

# display rows and columns of cleaned and encoded dataset
print ('='*80 + '\n')
print("Dataset No. of Rows: ", data.shape[0])
print("Dataset No. of Columns: ", data.shape[1])
print ('='*80 + '\n')

# Below are the commands that we referred to check structure, summary statistics of the cleaned dataset
# printing the dataset obseravtions
# print("Dataset first few rows:\n ")
# print(data.head(2))

# printing the struture of the dataset
# print("Dataset info:\n ")
# print(data.info())

# printing the summary statistics of the dataset
# print(data.describe(include='all'))


# In[3]:

# Extract features and labels
X = data.drop('price', axis = 1)
y = data['price']


# In[4]:

# perform training with random forest with all columns
# specify random forest Regressor
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=100)

# In[5]:

# Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# In[6]:

# perform training
clf.fit(X_train, y_train)

# In[7]:

# plot feature importances
# get feature importances
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, data.drop(columns = 'price').columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)

# show the plot
plt.tight_layout()
plt.show()


# In[8]:

# Lets review top 25 important features and their importance values
# print(f_importances.index[0:22]), print(f_importances.values[0:22])
print ('='*80 + '\n')
print("top 25 important features and their importance values")
print(f_importances.index[0:25], f_importances.values[0:25])
print ('='*80 + '\n')



# ==== Another way to cross check and Identify And Select Most Important Features with RandomForest Regressor ====

# In[9]:

 # selectFromModel from skleran automatically selects the good features and its good way to check
from sklearn.feature_selection import SelectFromModel

# In[10]:

# Create a selector object that will use the random forest regressor to identify features
select_features = SelectFromModel(RandomForestRegressor(n_estimators = 100))  # estimators are the number of trees
select_features.fit(X_train, y_train)


# In[11]:

# In order to check which features among all important we can use the method get_support()
select_features.get_support()

# This method will output an array of boolean values.
# True for the features whose importance is greater than the mean importance and False for the rest.

# In[12]:

# create list and count features
selected_feature= X_train.columns[(select_features.get_support())]
print("length of important features selected from RandomForestRegressor is:")
print(len(selected_feature))
print ('='*80 + '\n')

# In[13]:

# Display the names of the important features
print("list of selected features are",)
print(selected_feature)

# check and plot the distribution of importance.
f_importances.nlargest(22).plot(kind='barh')
plt.show()

# ### Above we have found the list of important features that are useful for us and impacting the pricing of the airbnb

# In[14]:

#From the above list of important features we will create another dataframe object with important features only
Regression_features = data.loc[:, ['host_listings_count', 'latitude', 'longitude', 'accommodates',
       'bathrooms', 'bedrooms', 'security_deposit', 'cleaning_fee',
       'guests_included', 'extra_people', 'availability_30', 'availability_60',
       'availability_90', 'availability_365', 'number_of_reviews',
       'number_of_reviews_ltm', 'review_scores_rating', 'reviews_per_month',
       'host_sinceyear', 'property_type_Hotel', 'room_type_Private room']]



# In[15]:

# placing target with extracted features for modeling
Regression_features = Regression_features.join(data['price'])
print ('='*80 + '\n')
print(Regression_features.head())
print ('='*80 + '\n')



# #  ================ Getting the best features from random forest into csv for Regression ===============

# In[16]:

for col in Regression_features.columns[Regression_features.isnull().any()]:
    print(col)

# generating csv
Regression_features.to_csv("Regression_features.csv", index=None)
print("csv created")
print(Regression_features.shape)
