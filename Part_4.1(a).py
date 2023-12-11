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
dev_set = pd.read_csv("my_dev.csv", sep = ',')


# In[3]:


# Train_Data
labels = train_data['SalePrice']
train_features = train_data.drop(['Id','SalePrice'],axis = 1)


# In[4]:


# Dev_set
d_labels = dev_set['SalePrice']
dev_features = dev_set.drop(['Id','SalePrice'], axis = 1)


# In[5]:


categorical_columns = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle','RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2','Heating', 'HeatingQC', 'CentralAir', 'Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']


# In[6]:


columns_to_replace_train = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']  
train_features[columns_to_replace_train] = train_features[columns_to_replace_train].fillna(0)


# In[7]:


columns_to_replace_dev = ['LotFrontage', 'MasVnrArea','GarageYrBlt']
dev_features[columns_to_replace_dev] = dev_features[columns_to_replace_dev].fillna(0)


# In[8]:


train_features[categorical_columns] = train_features[categorical_columns].astype(str)
dev_features[categorical_columns] = dev_features[categorical_columns].astype(str)


# In[9]:


#train = b.astype(str)
#devdata = d_features.astype(str)


# In[10]:


# Train Binary Conversion
encoder.fit(train_features)
bin_train = encoder.transform(train_features)

# Dev Binary conversion
bin_dev = encoder.transform(dev_features)


# In[11]:


X_train = bin_train
y_train = labels


# In[12]:


X_dev = bin_dev
y_dev = d_labels


# In[13]:


log_of_label = np.log(y_train)


# In[81]:


ridge_model = Ridge(alpha=20)


# In[82]:


ridge_model.fit(X_train, log_of_label)


# In[83]:


log_of_dev_label_ridge = ridge_model.predict(X_dev)


# In[84]:


final_price_predictions_ridge = np.exp(log_of_dev_label_ridge)
rmsle_ridge = np.sqrt(mean_squared_log_error(y_dev, final_price_predictions_ridge))


# In[85]:


rmsle_ridge


# In[ ]:





# In[ ]:




