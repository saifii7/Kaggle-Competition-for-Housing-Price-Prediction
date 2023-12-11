#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[2]:


train_data = pd.read_csv("my_train.csv", sep = ',')
test_set = pd.read_csv("test.csv", sep = ',')


# In[3]:


# Train_Data
labels = train_data['SalePrice']
train_features = train_data.drop(['Id','SalePrice'],axis = 1)


# test_Data
test_features = test_set.drop(['Id'],axis = 1)


# In[4]:


categorical_columns = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle','RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2','Heating', 'HeatingQC', 'CentralAir', 'Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']


# In[5]:


columns_to_replace_train = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']  
train_features[columns_to_replace_train] = train_features[columns_to_replace_train].fillna(0)


# In[6]:


columns_to_replace_test = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']  
test_features[columns_to_replace_test] = test_features[columns_to_replace_test].fillna(0)


# In[7]:


train_features[categorical_columns] = train_features[categorical_columns].astype(str)
test_features[categorical_columns] = test_features[categorical_columns].astype(str)


# In[9]:


# Train Binary Conversion
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(train_features)
bin_train = encoder.transform(train_features)

# Dev Binary conversion
bin_test = encoder.transform(test_features)


# In[10]:


X_train = bin_train
y_train = labels

X_test = bin_test


# In[11]:


log_of_label = np.log(y_train)


# In[12]:


ridge_model = Ridge(alpha=20)


# In[13]:


ridge_model.fit(X_train, log_of_label)


# In[19]:


log_of_test_label_ridge = ridge_model.predict(X_test)


# In[20]:


final_price_predictions_ridge = np.exp(log_of_test_label_ridge)


# In[22]:


submission_df = pd.DataFrame({'Id': test_set['Id'], 'SalePrice': final_price_predictions_ridge})


# In[23]:


submission_df.to_csv('Part4_naive.csv', index=False)


# In[ ]:




