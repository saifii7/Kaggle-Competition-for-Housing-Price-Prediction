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


# Train Data
labels = train_data['SalePrice']
train_features = train_data.drop(['Id', 'SalePrice'], axis=1)


# Dev Data
dev_labels = dev_set['SalePrice']
dev_features = dev_set.drop(['Id', 'SalePrice'], axis=1)


# In[4]:


columns_to_replace_train = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']  
train_features[columns_to_replace_train] = train_features[columns_to_replace_train].fillna(0)


# In[5]:


columns_to_replace_dev = ['LotFrontage', 'MasVnrArea','GarageYrBlt']
dev_features[columns_to_replace_dev] = dev_features[columns_to_replace_dev].fillna(0)


# In[6]:


numeric_columns = list(train_features.select_dtypes(include=[np.number]).columns)


# In[7]:


categorical_columns = list(train_features.select_dtypes(exclude=[np.number]).columns)


# In[8]:


column_to_move = 'MSSubClass'  
if column_to_move in numeric_columns:
    numeric_columns.remove(column_to_move)
    categorical_columns.append(column_to_move)
else:
    print("Column not found in numeric_columns list.")


# In[9]:


#train = train_features.astype(str)
#devdata = dev_features.astype(str)


# In[10]:


train_features[categorical_columns] = train_features[categorical_columns].astype(str)
dev_features[categorical_columns] = dev_features[categorical_columns].astype(str)


# In[11]:


num_processor = 'passthrough' # i.e., no transformation
cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[12]:


preprocessor = ColumnTransformer([
('num', num_processor, numeric_columns),
('cat', cat_processor, categorical_columns)])
preprocessor.fit(train_features)
train_processed_data = preprocessor.transform(train_features)
dev_processed_data = preprocessor.transform(dev_features)


# In[13]:


X_train = train_processed_data
y_train = labels


# In[14]:


X_dev = dev_processed_data
y_dev = dev_labels


# In[15]:


log_of_label = np.log(y_train)


# In[62]:


ridge_model = Ridge(alpha=20)


# In[63]:


ridge_model.fit(X_train, log_of_label)


# In[64]:


log_of_dev_label_ridge = ridge_model.predict(X_dev)


# In[65]:


final_price_predictions_ridge = np.exp(log_of_dev_label_ridge)
rmsle_ridge = np.sqrt(mean_squared_log_error(y_dev, final_price_predictions_ridge))


# In[66]:


rmsle_ridge


# In[ ]:




