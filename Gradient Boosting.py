#!/usr/bin/env python
# coding: utf-8

# In[15]:


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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


# In[16]:


train_data = pd.read_csv("my_train.csv", sep = ',')
test_set = pd.read_csv("test.csv", sep = ',')


# In[17]:


# Train_Data
labels = train_data['SalePrice']
train_features = train_data.drop(['Id','SalePrice'],axis = 1)


# test_Data
test_features = test_set.drop(['Id'],axis = 1)


# In[18]:


columns_to_replace_train_features = ['LotFrontage', 'MasVnrArea','GarageYrBlt']
train_features[columns_to_replace_train_features] = train_features[columns_to_replace_train_features].fillna(0)


# In[19]:


columns_to_replace_test = ['LotFrontage', 'MasVnrArea','BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea','GarageYrBlt']
test_features[columns_to_replace_test] = test_features[columns_to_replace_test].fillna(0)


# In[20]:


numeric_columns = list(train_features.select_dtypes(include=[np.number]).columns)
categorical_columns = list(train_features.select_dtypes(exclude=[np.number]).columns)


# In[21]:


column_to_move = 'MSSubClass'  
if column_to_move in numeric_columns:
    numeric_columns.remove(column_to_move)
    categorical_columns.append(column_to_move)
else:
    print("Column not found in numeric_columns list.")


# In[22]:


train_features[categorical_columns] = train_features[categorical_columns].astype(str)
test_features[categorical_columns] = test_features[categorical_columns].astype(str)


# In[23]:


num_processor = 'passthrough' # i.e., no transformation
cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[24]:


preprocessor = ColumnTransformer([
('num', num_processor, numeric_columns),
('cat', cat_processor, categorical_columns)])
preprocessor.fit(train_features)
train_processed_data = preprocessor.transform(train_features)
test_processed_data = preprocessor.transform(test_features)


# In[25]:


X_train = train_processed_data
y_train = labels


X_test = test_processed_data


# In[26]:


important_features = ['GarageArea','KitchenAbvGr','BsmtFullBath','1stFlrSF','BsmtFinSF1','YearRemodAdd','YearBuilt','LotFrontage','LotArea','OverallQual','OverallCond']  
important_numeric_columns = [col for col in numeric_columns if col in important_features]

poly = PolynomialFeatures(degree=2, include_bias=False)
train_poly_features = poly.fit_transform(train_features[important_numeric_columns])
test_poly_features = poly.transform(test_features[important_numeric_columns])


X_train_poly = np.concatenate((X_train, train_poly_features), axis=1)
X_test_poly = np.concatenate((X_test, test_poly_features), axis=1)


# In[27]:


log_of_label = np.log(y_train)


# In[28]:


gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gradient_boosting_model.fit(X_train_poly, log_of_label)
log_of_test_label = gradient_boosting_model.predict(X_test_poly)
final_price_predictions = np.exp(log_of_test_label)
#rmsle = np.sqrt(mean_squared_log_error(y_dev, final_price_predictions))


# In[29]:


submission_df = pd.DataFrame({'Id': test_set['Id'], 'SalePrice': final_price_predictions})


# In[30]:


submission_df.to_csv('Gradient_Boosting.csv', index=False)


# In[ ]:




