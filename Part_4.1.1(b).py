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


# In[23]:


# Train_Data
labels = train_data['SalePrice']
train_features = train_data.drop(['Id','SalePrice'],axis = 1)


# test_Data
test_features = test_set.drop(['Id'],axis = 1)


# In[4]:


columns_to_replace_train_features = ['LotFrontage', 'MasVnrArea','GarageYrBlt']
train_features[columns_to_replace_train_features] = train_features[columns_to_replace_train_features].fillna(0)


# In[5]:


columns_to_replace_test = ['LotFrontage', 'MasVnrArea','BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea','GarageYrBlt']
test_features[columns_to_replace_test] = test_features[columns_to_replace_test].fillna(0)


# In[6]:


numeric_columns = list(train_features.select_dtypes(include=[np.number]).columns)
categorical_columns = list(train_features.select_dtypes(exclude=[np.number]).columns)


# In[7]:


column_to_move = 'MSSubClass'  
if column_to_move in numeric_columns:
    numeric_columns.remove(column_to_move)
    categorical_columns.append(column_to_move)
else:
    print("Column not found in numeric_columns list.")


# In[8]:


train_features[categorical_columns] = train_features[categorical_columns].astype(str)
test_features[categorical_columns] = test_features[categorical_columns].astype(str)


# In[9]:


num_processor = 'passthrough' # i.e., no transformation
cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[10]:


preprocessor = ColumnTransformer([
('num', num_processor, numeric_columns),
('cat', cat_processor, categorical_columns)])
preprocessor.fit(train_features)
train_processed_data = preprocessor.transform(train_features)
test_processed_data = preprocessor.transform(test_features)


# In[11]:


X_train = train_processed_data
y_train = labels

X_test = test_processed_data


# In[12]:


log_of_label = np.log1p(y_train)


# In[13]:


ridge_model = Ridge(alpha=20)


# In[14]:


ridge_model.fit(X_train, log_of_label)


# In[19]:


log_of_test_label_ridge = ridge_model.predict(X_test)


# In[20]:


final_price_predictions_ridge = np.exp(log_of_test_label_ridge)


# In[21]:


submission_df = pd.DataFrame({'Id': test_set['Id'], 'SalePrice': final_price_predictions_ridge})


# In[22]:


submission_df.to_csv('Part4_smart.csv', index=False)


# In[ ]:




