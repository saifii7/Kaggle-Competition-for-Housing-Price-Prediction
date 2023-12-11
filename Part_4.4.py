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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


# In[2]:


train_data = pd.read_csv("my_train.csv", sep = ',')
dev_set = pd.read_csv("my_dev.csv", sep = ',')


# In[3]:


# Train_Data
labels = train_data['SalePrice']
train_features = train_data.drop(['Id','SalePrice'],axis = 1)


# In[4]:


# Dev_set
dev_labels = dev_set['SalePrice']
dev_features = dev_set.drop(['Id','SalePrice'], axis = 1)


# In[5]:


columns_to_replace_train = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']  
train_features[columns_to_replace_train] = train_features[columns_to_replace_train].fillna(0)


# In[6]:


columns_to_replace_dev = ['LotFrontage', 'MasVnrArea','GarageYrBlt']
dev_features[columns_to_replace_dev] = dev_features[columns_to_replace_dev].fillna(0)


# In[7]:


numeric_columns = list(train_features.select_dtypes(include=[np.number]).columns)
categorical_columns = list(train_features.select_dtypes(exclude=[np.number]).columns)


# In[8]:


column_to_move = 'MSSubClass'  
if column_to_move in numeric_columns:
    numeric_columns.remove(column_to_move)
    categorical_columns.append(column_to_move)
else:
    print("Column not found in numeric_columns list.")


# In[9]:


train_features[categorical_columns] = train_features[categorical_columns].astype(str)
dev_features[categorical_columns] = dev_features[categorical_columns].astype(str)


# In[10]:


num_processor = 'passthrough' # i.e., no transformation
cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[11]:


preprocessor = ColumnTransformer([
('num', num_processor, numeric_columns),
('cat', cat_processor, categorical_columns)])
preprocessor.fit(train_features)
train_processed_data = preprocessor.transform(train_features)
dev_processed_data = preprocessor.transform(dev_features)


# In[12]:


X_train = train_processed_data
y_train = labels


# In[13]:


X_dev = dev_processed_data
y_dev = dev_labels


# In[14]:



important_features = ['GarageYrBlt','KitchenAbvGr','BsmtHalfBath','MSSubClass','WoodDeckSF','GarageArea','KitchenAbvGr','BsmtFullBath','1stFlrSF','BsmtFinSF1','YearRemodAdd','YearBuilt','LotFrontage','LotArea','OverallQual','OverallCond'] 
important_numeric_columns = [col for col in numeric_columns if col in important_features]

poly = PolynomialFeatures(degree=2, include_bias=False)
train_poly_features = poly.fit_transform(train_features[important_numeric_columns])
dev_poly_features = poly.transform(dev_features[important_numeric_columns])

X_train_poly = np.concatenate((X_train, train_poly_features), axis=1)
X_dev_poly = np.concatenate((X_dev, dev_poly_features), axis=1)

log_of_label = np.log(y_train)

# Ridge Regression model
#alpha = 20  
#ridge_model = Ridge(alpha=alpha)
#ridge_model.fit(X_train_poly, log_of_label)
#log_of_dev_label = ridge_model.predict(X_dev_poly)
#final_price_predictions = np.exp(log_of_dev_label)
#rmsle = np.sqrt(mean_squared_log_error(y_dev, final_price_predictions))



# Gradient Boosting Regression model
#gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
#gradient_boosting_model.fit(X_train_poly, log_of_label)
#log_of_dev_label = gradient_boosting_model.predict(X_dev_poly)
#final_price_predictions = np.exp(log_of_dev_label)
#rmsle = np.sqrt(mean_squared_log_error(y_dev, final_price_predictions))




# SVR model
#svr_model = SVR(kernel='rbf')  
#svr_model.fit(X_train_poly, log_of_label)
#log_of_dev_label = svr_model.predict(X_dev_poly)
#final_price_predictions = np.exp(log_of_dev_label)
#rmsle = np.sqrt(mean_squared_log_error(y_dev, final_price_predictions))




# Random Forest Regression model
n_estimators = 100  # Number of trees in the forest
max_depth = 10  # Maximum depth of the trees
rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
rf_model.fit(X_train_poly, log_of_label)
log_of_dev_label = rf_model.predict(X_dev_poly)
final_price_predictions = np.exp(log_of_dev_label)
rmsle = np.sqrt(mean_squared_log_error(y_dev, final_price_predictions))





#model = LinearRegression()
#model.fit(X_train_poly, log_of_label)


rmsle


# In[ ]:





# In[ ]:




