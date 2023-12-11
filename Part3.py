#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
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
categorical_columns = list(train_features.select_dtypes(exclude=[np.number]).columns)


# In[7]:


column_to_move = 'MSSubClass'  
if column_to_move in numeric_columns:
    numeric_columns.remove(column_to_move)
    categorical_columns.append(column_to_move)
else:
    print("Column not found in numeric_columns list.")


# In[10]:


train_features[categorical_columns] = train_features[categorical_columns].astype(str)
dev_features[categorical_columns] = dev_features[categorical_columns].astype(str)


# In[11]:


train_features.shape


# In[12]:


#train = train_features.astype(str)
#devdata = dev_features.astype(str)


# In[13]:


num_processor = 'passthrough' # i.e., no transformation
cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[14]:


preprocessor = ColumnTransformer([
('num', num_processor, numeric_columns),
('cat', cat_processor, categorical_columns)])
preprocessor.fit(train_features)
train_processed_data = preprocessor.transform(train_features)
dev_processed_data = preprocessor.transform(dev_features)


# In[15]:


X_train = train_processed_data
y_train = labels


# In[16]:


X_dev = dev_processed_data
y_dev = dev_labels


# In[17]:


log_of_label = np.log(y_train)


# In[18]:


model = LinearRegression()
model.fit(X_train,log_of_label)


# In[19]:


log_of_dev_label = model.predict(X_dev)


# In[20]:


final_price_predictions = np.exp(log_of_dev_label)


# In[21]:


rmsle = np.sqrt(mean_squared_log_error(y_dev,final_price_predictions))


# In[22]:


rmsle


# In[23]:


#3(a)
train_processed_data.shape


# In[24]:


coef = model.coef_


# In[25]:


f_names = preprocessor.get_feature_names_out()


# In[26]:


coef_feature_pairs = list(zip(f_names,coef))


# In[27]:


sorted_pairs = sorted(coef_feature_pairs, key=lambda x:x[1], reverse=True)


# In[28]:


sorted_pairs


# In[29]:


# Extract the top 10 most positive features
top_positive_features = sorted_pairs[:10]


# In[30]:


top_positive_features


# In[31]:


# Extract the top 10 most negative features
top_negative_features = sorted_pairs[-10:]


# In[32]:


top_negative_features = sorted(sorted_pairs[-10:], key=lambda x: x[1], reverse=False)


# In[33]:


top_negative_features


# In[34]:


# Access the bias term (intercept)
bias_term = model.intercept_


# In[35]:


bias_term


# In[ ]:





# In[ ]:




