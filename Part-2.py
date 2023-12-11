#!/usr/bin/env python
# coding: utf-8

# In[38]:


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


# In[39]:


train_data = pd.read_csv("my_train.csv", sep = ',')
dev_set = pd.read_csv("my_dev.csv", sep = ',')


# In[40]:


# Train_Data
labels = train_data['SalePrice']
train_features = train_data.drop(['Id','SalePrice'],axis = 1)


# In[41]:


# Dev_set
d_labels = dev_set['SalePrice']
dev_features = dev_set.drop(['Id','SalePrice'], axis = 1)


# In[42]:


categorical_columns = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle','RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2','Heating', 'HeatingQC', 'CentralAir', 'Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']


# In[43]:


columns_to_replace_train = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']  
train_features[columns_to_replace_train] = train_features[columns_to_replace_train].fillna(0)


# In[45]:


train_features[categorical_columns] = train_features[categorical_columns].astype(str)
dev_features[categorical_columns] = dev_features[categorical_columns].astype(str)


# In[46]:


train_features.shape


# In[47]:


# Train Binary Conversion
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(train_features)
bin_train = encoder.transform(train_features)

# Dev Binary conversion
bin_dev = encoder.transform(dev_features)


# In[48]:


X_train = bin_train
y_train = labels


# In[49]:


X_dev = bin_dev
y_dev = d_labels


# In[50]:


log_of_label = np.log(y_train)


# In[51]:


model = LinearRegression()
model.fit(X_train,log_of_label)


# In[52]:


log_of_dev_label = model.predict(X_dev)


# In[53]:


final_price_predictions = np.exp(log_of_dev_label)


# In[54]:


rmsle = np.sqrt(mean_squared_log_error(y_dev,final_price_predictions))


# In[55]:


rmsle


# In[56]:


coef = model.coef_


# In[57]:


f_names = encoder.get_feature_names_out()


# In[58]:


f_names


# In[59]:


coef_feature_pairs = list(zip(f_names,coef))


# In[60]:


coef_feature_pairs


# In[61]:


sorted_pairs = sorted(coef_feature_pairs, key=lambda x:x[1], reverse=True)


# In[62]:


sorted_pairs


# In[63]:


# Extract the top 10 most positive features
top_positive_features = sorted_pairs[:10]


# In[64]:


top_positive_features


# In[65]:


# Extract the top 10 most positive features
top_negative_features = sorted_pairs[-10:]


# In[66]:


top_negative_features = sorted(sorted_pairs[-10:], key=lambda x: x[1], reverse=False)


# In[67]:


top_negative_features


# In[68]:


# Access the bias term (intercept)
bias_term = model.intercept_


# In[69]:


bias_term


# In[ ]:




