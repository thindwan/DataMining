
# # =========================== Data Modeling ==================================


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

# In[1]: --------------------------------------

# import Dataset
data2 = pd.read_csv("Regression_features.csv")

# In[2]:

print(data2.head())

## ===================== Correlation Matrix ====================

# In[3]:

plt.figure(figsize=(30, 20))
sns.heatmap(data2.corr(), annot=True, linewidths=0.1, cmap='Reds')
plt.show()

# ### We can see above that there are very few features out of f_important that are have positive correlation or influence.
# Features like ['accommodates','bathrooms', 'bedrooms', 'security_deposit', 'cleaning_fee', 'guests_included','availability_365','property_type_Hote]'
# are positively correlated. So we will try to do regression first using positively correlated features from Correlation matrix and then secondly from the randomforest for our reference.
#### For our dataset, we are considering correlation values between .20 to .5 as good.

# In[4]:

# lets separate positively correlated features and plot for better visualization for our ease
coormatrix_features = data2.loc[:, ('accommodates', 'bathrooms', 'bedrooms', 'security_deposit',
                                    'cleaning_fee', 'guests_included', 'availability_365', 'property_type_Hotel',
                                    'price')]

# In[5]:

plt.figure(figsize=(30, 20))
sns.heatmap(coormatrix_features.corr(), annot=True, linewidths=0.1, cmap='Reds')
plt.show()

# # ======== Linear Regression using features from Coorelation Matrix ==========
# In[6]:

# Get Features and Target
U = coormatrix_features.drop('price', axis=1)
V = coormatrix_features['price']

# In[7]:

# split the dataset into train and test
from sklearn.model_selection import train_test_split

U_train, U_test, V_train, V_test = train_test_split(U, V, test_size=0.2, random_state=10)

# In[8]:
# Standardize the features and target / # normalizing the features target

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

U_train = ss.fit_transform(U_train)
U_test = ss.transform(U_test)  # borrowing parameters from train
print(U_train.shape, U_test.shape)
V_train = ss.fit_transform(V_train.values.reshape(-1, 1))
V_test = ss.transform(V_test.values.reshape(-1, 1))

print(V_train.shape, V_test.shape)

# In[9]:
# ------Apply Linear Model-----------------
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

regr = linear_model.LinearRegression()
regr.fit(U_train, V_train)
airbnb_V_pred = regr.predict(U_test)
print('#', 50 * "-")
print('Coefficients: \n', regr.coef_)
print('#', 50 * "-")
print("Mean squared error: %.2f"
      % mean_squared_error(V_test, airbnb_V_pred))
print('#', 50 * "-")
print('R2 score : %.2f' % r2_score(V_test, airbnb_V_pred))
print('#', 50 * "-")
print('SCORE OF THE MODEL: ', regr.score(U_test, V_test))
print('#', 50 * "-")

# In[10]:


# For Vizualization between actual prices and predicted values we can create the scatter plot
plt.scatter(V_test, airbnb_V_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()

#### There should be linear straight line. But we can see that actual values are quite far apart from the predicted values.

## --------------- Simple Linear Regression with all the features selected from RandomForest Feature Selection ---------

# In[11]:

# Extract features and labels
X = data2.drop('price', axis=1)
y = data2['price']

# split the dataset into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# In[12]:

# Standardize the features and target / # normalizing the features target
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)  # borrowing parameters from train
print(X_train.shape, X_test.shape)

y_train = ss.fit_transform(y_train.values.reshape(-1, 1))
y_test = ss.transform(y_test.values.reshape(-1, 1))

print(y_train.shape, y_test.shape)

# In[13]:
# -----------Apply linear model------------
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
airbnb_y_pred = regr.predict(X_test)

# In[14]:

# For Vizualization between actual prices abd predicted values we can create the scatter plot again
plt.scatter(y_test, airbnb_y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()

# ### The plot is almost same as before.
# ### We can see that there is no such linear line so the model is not fitting 100%. But for our kind of
# project where we have social science data, having user described information we can consider it as good.
#### scatter plot should create a linear line. As the model does not fit 100%, the scatter plot is not creating a linear line

# In[15]:

# Performance Metrics
print('#', 50 * "-")
print('Coefficients: \n', regr.coef_)
print('#', 50 * "-")
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, airbnb_y_pred))
print('#', 50 * "-")
print('R2 score: %.2f' % r2_score(y_test, airbnb_y_pred))
print('#', 50 * "-")
print('SCORE OF THE MODEL: ', regr.score(X_test, y_test))
print('#', 50 * "-")

# In[16]:

print(
    ' Determination Coefficent values for our kind of social science data with 0.35 R2 value and score 0.35 and MSE: 0.62 can be considered as good')



