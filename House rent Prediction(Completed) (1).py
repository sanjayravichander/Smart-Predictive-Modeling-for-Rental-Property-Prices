#!/usr/bin/env python
# coding: utf-8

# In[1]:


## House Rent Prediction Train Dataset
import pandas as pd
rent_train=pd.read_excel("C:\\Users\\DELL\\Downloads\\House Rent\\House_Rent_Train.xlsx")


# In[2]:


## House Rent Prediction Test Dataset
import pandas as pd
rent_test=pd.read_excel("C:\\Users\\DELL\\Downloads\\House Rent\\House_Rent_Test.xlsx")


# In[3]:


## Converting the values in type as It is showing the same values in uppercase and lowercase(Train Dataset)
rent_train['type'] = rent_train['type'].str.upper()

# Replacing occurrences of "1BHK1" with "BHK1"
rent_train['type'] = rent_train['type'].replace('1BHK1', 'BHK1')


# In[4]:


# Extract main area from 'locality' column Train Dataset
rent_train['locality'] = rent_train['locality'].str.split(',').str[0]


# In[5]:


# Extract main area from 'locality' column Test Dataset
rent_test['locality'] = rent_test['locality'].str.split(',').str[0]


# In[6]:


rent_train.shape


# In[7]:


rent_train.isnull().sum()


# In[8]:


rent_test.isnull().sum()


# In[6]:


from sklearn.impute import SimpleImputer #Train Dataset

# List of categorical features
categorical_features = ['type', 'locality', 'lease_type', 'building_type', 'water_supply', 'facing']

# Create an imputer object
imputer = SimpleImputer(strategy='most_frequent')

# Loop over each categorical feature and fill missing values
for feature in categorical_features:
    # Extract the feature as a DataFrame
    feature_df = pd.DataFrame(rent_train[feature])
    
    # Fill missing values in the feature DataFrame
    filled_values = imputer.fit_transform(feature_df)
    
    # Replace the original feature with the filled values
    rent_train[feature] = pd.DataFrame(filled_values)


# In[7]:


# Convert 'activation_date' to datetime format, ignoring errors(Train Dataset)
rent_train['activation_date'] = pd.to_datetime(rent_train['activation_date'],dayfirst=True, errors='coerce')

# Impute missing values in 'activation_date' with the most frequent date
most_frequent_date = rent_train['activation_date'].mode()[0]
rent_train['activation_date'].fillna(most_frequent_date, inplace=True)


# In[8]:


from sklearn.impute import SimpleImputer#(Train Dataset)

# Define columns with missing values
columns_with_missing = ['property_age', 'bathroom', 'cup_board', 'floor', 'total_floor', 'balconies', 'rent']

# Create an imputer object for numerical features (using mean strategy)
num_imputer = SimpleImputer(strategy='mean')

# Impute missing values in numerical features
rent_train[columns_with_missing] = num_imputer.fit_transform(rent_train[columns_with_missing])


# In[9]:


import pandas as pd
import json
from pandas import json_normalize

# Sample DataFrame
data_train = rent_train['amenities']
df_train = pd.DataFrame(data_train)

# Function to parse string representation of dictionaries
def parse_amenities_string(s):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}

# Convert string representation of dictionaries to actual dictionaries
df_train['amenities'] = df_train['amenities'].apply(parse_amenities_string)

# Normalize the 'amenities' column into separate columns
df_train = pd.json_normalize(df_train['amenities'])

# Fill null values with False (or any other desired value)
df_train = df_train.fillna(False)


# In[10]:


rent_train=pd.concat([rent_train,df_train],axis=1)


# In[11]:


# Amenities on Test Dataset
# Sample DataFrame
data_test = rent_test['amenities']
df_test = pd.DataFrame(data_test)

# Convert string representation of dictionaries to actual dictionaries
df_test['amenities'] = df_test['amenities'].apply(parse_amenities_string)

# Normalize the 'amenities' column into separate columns
df_test = pd.json_normalize(df_test['amenities'])

# Fill null values with False (or any other desired value)
df_test = df_test.fillna(False)


# In[12]:


rent_test=pd.concat([rent_test,df_test],axis=1)


# In[13]:


# Remove duplicate rows from the Train dataset
rent_train_duplicates = rent_train.drop_duplicates()

# Verify if duplicates are removed
print("Original train dataset length:", len(rent_train))
print("After removing Duplicates:", len(rent_train_duplicates))


# In[14]:


# Remove duplicate rows from the Test dataset
rent_test_duplicates = rent_test.drop_duplicates()

# Verify if duplicates are removed
print("Original test dataset length:", len(rent_test))
print("After removing Duplicates:", len(rent_test_duplicates))


# In[15]:


#Dropping columns from training dataset
rent_train.drop(['LIFT','GYM','POOL','FS','SC','GP','PARK','PB','VP'],axis=1,inplace=True)


# In[16]:


#Dropping columns from testing dataset
rent_test.drop(['LIFT','GYM','POOL','FS','SC','GP','PARK','PB','VP'],axis=1,inplace=True)


# In[17]:


rent_train.drop(['amenities'],axis=1,inplace=True)
rent_test.drop(['amenities'],axis=1,inplace=True)


# In[18]:


# Define a dictionary to map old column names to new column names
column_mapping = {
    'AC': 'Air_Conditioning',
    'CLUB': 'Clubhouse',
    'INTERCOM': 'Intercom',
    'SERVANT': 'Servant_Quarters',
    'SECURITY': 'Security_System',
    'RWH': 'Rainwater_Harvesting',
    'STP': 'Sewage_Treatment_Plant',
    'HK': 'Housekeeping'
}

# Rename columns using the dictionary
rent_train.rename(columns=column_mapping, inplace=True)
rent_test.rename(columns=column_mapping, inplace=True)


# In[19]:


rent_train.drop(['id'],axis=1,inplace=True)
rent_test.drop(['id'],axis=1,inplace=True)


# In[20]:


rent_train['locality'] = rent_train['locality'].str.replace(' ', '')
rent_test['locality'] = rent_test['locality'].str.replace(' ', '')


# In[21]:


#For training Dataset
location_status_train=rent_train.groupby('locality')['locality'].agg('count').sort_values(ascending=False)
location_status_train


# In[25]:


len(location_status_train[location_status_train<10])


# In[24]:


location_status_train_less_than_10=location_status_train[location_status_train<10]
location_status_train_less_than_10


# In[25]:


rent_train['locality']=rent_train['locality'].apply(lambda x: 'other' if x in location_status_train_less_than_10 else x)


# In[26]:


# For Testing Dataset
location_status_test=rent_test.groupby('locality')['locality'].agg('count').sort_values(ascending=False)
location_status_test


# In[27]:


len(location_status_test[location_status_test<10])


# In[28]:


location_status_test_less_than_10=location_status_test[location_status_test<10]
location_status_test_less_than_10


# In[30]:


rent_test['locality']=rent_test['locality'].apply(lambda x: 'other' if x in location_status_test_less_than_10 else x)


# In[31]:


# Dropping from train and test datasets
rent_train.drop(['Clubhouse','CPA'],axis=1,inplace=True)
rent_train.drop(['latitude','longitude'],axis=1,inplace=True)

rent_test.drop(['Clubhouse','CPA'],axis=1,inplace=True)
rent_test.drop(['latitude','longitude'],axis=1,inplace=True)


# In[33]:


# Outliers for training dataset
# Selecting numerical columns excluding 'gym', 'lift', 'swimming_pool', and 'negotiable'
numerical_columns_train = [col for col in rent_train.select_dtypes(include=['float64', 'int64']).columns if col not in ['gym', 'lift', 'swimming_pool', 'negotiable']]

import plotly.graph_objects as go

# Creating box plots for each numerical column
data_train = []
for column in numerical_columns_train:
    data_train.append(go.Box(y=rent_train[column].dropna(), name=column, boxmean=True))

# Creating layout
layout = go.Layout(
    title='Outliers Chart',
    xaxis=dict(title='Numerical Columns'),
    yaxis=dict(title='Values'),
    boxmode='group'  # Show box plots side-by-side
)

# Creating figure
fig = go.Figure(data=data_train, layout=layout)

# Displaying the interactive plot
fig.show()


# In[34]:


rent_test['activation_date']=pd.to_datetime(rent_test['activation_date'],dayfirst=True)


# In[35]:


# Outliers for testing dataset
# Selecting numerical columns excluding 'gym', 'lift', 'swimming_pool', and 'negotiable'
numerical_columns_test = [col for col in rent_test.select_dtypes(include=['float64', 'int64']).columns if col not in ['gym', 'lift', 'swimming_pool', 'negotiable']]

# Creating box plots for each numerical column
data_test = []
for column in numerical_columns_test:
    data_test.append(go.Box(y=rent_test[column].dropna(), name=column, boxmean=True))

# Creating layout
layout = go.Layout(
    title='Outliers Chart',
    xaxis=dict(title='Numerical Columns'),
    yaxis=dict(title='Values'),
    boxmode='group'  # Show box plots side-by-side
)

# Creating figure
fig_test = go.Figure(data=data_test, layout=layout)

# Displaying the interactive plot
fig_test.show()


# In[36]:


rent_train['property_size']


# In[34]:


# Handling Outliers for Training Dataset
def handle_outliers_train(df_train, column):
    Q1 = df_train[column].quantile(0.25)
    Q3 = df_train[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound_train = Q1 - 1.5 * IQR
    upper_bound_train = Q3 + 1.5 * IQR
 
    df_train[column] = df_train[column].apply(lambda x: upper_bound_train if x > upper_bound_train else lower_bound_train if x < lower_bound_train else x)
    
    return df_train

# Using the function on columns
rent_train = handle_outliers_train(rent_train, 'rent')
rent_train = handle_outliers_train(rent_train, 'property_size')


# In[35]:


# Handling Outliers for Testing Dataset
def handle_outliers_test(df_test, column_test):
    Q1_test = df_test[column_test].quantile(0.25)
    Q3_test = df_test[column_test].quantile(0.75)
    IQR_test = Q3_test - Q1_test

    lower_bound_test = Q1_test - 1.5 * IQR_test
    upper_bound_test = Q3_test + 1.5 * IQR_test

    df_test[column_test] = df_test[column_test].apply(lambda x: upper_bound_test if x > upper_bound_test else lower_bound_test if x < lower_bound_test else x)
    
    return df_test

# Using the function on columns
rent_test = handle_outliers_test(rent_test, 'property_age')
rent_test = handle_outliers_test(rent_test, 'property_size')


# In[36]:


# Removing from training and testing dataset
rent_train.drop(['negotiable','property_age','cup_board','floor'],axis=1,inplace=True)
rent_test.drop(['negotiable','property_age','cup_board','floor'],axis=1,inplace=True)


# In[40]:


# Correlation for training dataset
import plotly.io as pio

# Selecting numerical columns
numerical_columns = rent_train.select_dtypes(include=['float64', 'int64']).columns

# Calculating the correlation matrix with 'rent_log' as the target column
correlation_matrix = rent_train[numerical_columns].corrwith(rent_train['rent'])

# Plotting the correlation heatmap
data = go.Heatmap(z=correlation_matrix.values.reshape(1, -1),
                  x=correlation_matrix.index,
                  y=['Rent'],
                  colorscale='RdBu',
                  zmin=-1,
                  zmax=1,
                  hovertemplate='Feature: %{x}<br>Correlation: %{z:.2f}')

# Creating layout
layout = go.Layout(title='Correlation Heatmap with Rent Log',
                   xaxis=dict(title='Features'),
                   yaxis=dict(title='Rent'))

# Creating figure
fig = go.Figure(data=[data], layout=layout)

# Displaying the interactive heatmap
pio.show(fig)


# In[41]:


## Distributions of Categorical Values for Training Dataset

# Selecting categorical columns excluding 'id' and 'locality'
categorical_columns = [col for col in rent_train.select_dtypes(include=['object']).columns if col not in ['id', 'locality']]

# Creating subplots
fig = go.Figure()

# Plotting bar charts for each categorical column
for column in categorical_columns:
    category_counts = rent_train[column].value_counts()
    fig.add_trace(go.Bar(x=category_counts.index.astype(str), y=category_counts.values, name=column))

# Customizing layout
fig.update_layout(
    title='Distribution of Categorical Columns',
    xaxis=dict(title='Categories'),
    yaxis=dict(title='Frequency'),
    barmode='group'  # Show bars side-by-side
)

# Displaying the interactive plot
pio.show(fig)


# In[37]:


#ANOVA test for category columns for getting imp columns
import pandas as pd
from scipy.stats import f_oneway

# Selecting categorical columns excluding 'id' and 'locality'
categorical_columns = [col for col in rent_train.select_dtypes(include=['object','bool']).columns if col not in ['id', 'locality']]

# Calculating ANOVA F-statistic for each categorical column
anova_results = {}
for column in categorical_columns:
    groups = []
    for category in rent_train[column].unique():
        groups.append(rent_train[rent_train[column] == category]['rent'])
    anova_results[column] = f_oneway(*groups).statistic

# Printing ANOVA F-statistic for each categorical column
for column, f_statistic in anova_results.items():
    print(f"ANOVA F-statistic for {column}: {f_statistic}")


# In[38]:


# Removing columns from training and testing
rent_train.drop(['INTERNET','facing'],axis=1,inplace=True)
rent_test.drop(['INTERNET','facing'],axis=1,inplace=True)


# In[39]:


import category_encoders as ce

# Define the categorical columns to be encoded
categorical_columns = ['furnishing', 'parking', 'water_supply', 'building_type', 'lease_type', 'type']

# Initialize target encoders for each categorical column
target_encoders = {}
for column in categorical_columns:
    target_encoders[column] = ce.TargetEncoder(cols=[column])

# Fit and transform the encoders on the training data
encoded_columns_train = []
for column, encoder in target_encoders.items():
    encoded_column_train = encoder.fit_transform(rent_train[column], rent_train['rent'])
    encoded_columns_train.append(encoded_column_train)

# Concatenate the encoded columns with the original training DataFrame
rent_train = pd.concat([rent_train] + encoded_columns_train, axis=1)

# Apply the same encoders to the testing data
encoded_columns_test = []
for column, encoder in target_encoders.items():
    encoded_column_test = encoder.transform(rent_test[column])
    encoded_columns_test.append(encoded_column_test)

# Concatenate the encoded columns with the original testing DataFrame
rent_test = pd.concat([rent_test] + encoded_columns_test, axis=1)

# Drop the original categorical columns
rent_train = rent_train.drop(columns=categorical_columns)
rent_test = rent_test.drop(columns=categorical_columns)


# In[40]:


import category_encoders as ce

# Initialize target encoder for the 'locality' column
locality_target_encoder = ce.TargetEncoder(cols=['locality'])

# Fit and transform the target encoder on the training data
rent_train['Locality'] = locality_target_encoder.fit_transform(rent_train['locality'], rent_train['rent'])

# Map the encoded values to the 'locality' column in the testing dataset
rent_test['Locality'] = locality_target_encoder.transform(rent_test['locality'])

# Drop the original 'locality' column from both datasets
rent_train.drop(columns=['locality'], inplace=True)
rent_test.drop(columns=['locality'], inplace=True)


# In[41]:


# Converting 'activation_date' column to datetime type if it's not already in the training dataset
rent_train['activation_date'] = pd.to_datetime(rent_train['activation_date'])

# Extract year and month from 'activation_date' in the training dataset
rent_train['activation_year'] = rent_train['activation_date'].dt.year
rent_train['activation_month'] = rent_train['activation_date'].dt.month

# Drop the original 'activation_date' column from the training dataset
rent_train.drop(columns=['activation_date'], inplace=True)

# Apply the same transformations to the testing dataset
rent_test['activation_date'] = pd.to_datetime(rent_test['activation_date'])
rent_test['activation_year'] = rent_test['activation_date'].dt.year
rent_test['activation_month'] = rent_test['activation_date'].dt.month
rent_test.drop(columns=['activation_date'], inplace=True)


# In[43]:


# Encoding labelling in both train and test
from sklearn.preprocessing import LabelEncoder

# Define the columns to be encoded
columns_to_encode = ['Air_Conditioning', 'Intercom', 'Servant_Quarters', 'Security_System',
                     'Rainwater_Harvesting', 'Sewage_Treatment_Plant', 'Housekeeping']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to each column in the training dataset
for column in columns_to_encode:
    rent_train[column] = label_encoder.fit_transform(rent_train[column])

# Apply the same transformations to the testing dataset
for column in columns_to_encode:
    rent_test[column] = label_encoder.transform(rent_test[column])


# In[44]:


# Drop the two columns from the original DataFrame and save the result in a new DataFrame
rent_train_1 = rent_train.drop(['rent'], axis=1).copy()


# In[45]:


rent_train_2=rent_train['rent'].copy()


# In[50]:


rent_train_1.head(2)


# In[51]:


rent_test.head(2)


# In[46]:


# Step 2: Train-Test Split (for model evaluation during training)
Inputs=rent_train_1
Output=rent_train_2


# In[47]:


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(Inputs, Output, test_size=0.21888, random_state=42)


# In[48]:


import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define a function for inverse log transformation
#def inverse_log_transform(y_log):
    #return np.exp(y_log)

# Apply inverse log transformation to target variables
#y_train_original_scale = inverse_log_transform(y_train)
#y_val_original_scale = inverse_log_transform(y_val)

# Define regression models and their respective hyperparameter grids
Regression_Models = {
    'Linear Regression': (LinearRegression(), {}),
    'Ridge Regression': (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
    'Lasso Regression': (Lasso(), {'alpha': [0.1, 1.0, 10.0]}),
    'ElasticNet Regression': (ElasticNet(), {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}),
    'Bayesian Ridge Regression': (BayesianRidge(), {}),
    'Decision Tree': (DecisionTreeRegressor(), {'criterion': ['squared_error', 'absolute_error', 'poisson', 'friedman_mse'], 'max_depth': [None, 5, 10]}),
    'Random Forest': (RandomForestRegressor(), {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}),
    'XG Boost': (XGBRegressor(), {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01], 'max_depth': [3, 5, 7]}),
}

# Loop over models
for model_name, (model, param_grid) in Regression_Models.items():
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(x_train, y_train.values.ravel())  # Convert y_train to a 1D array using ravel()
    
    # Predict on validation set
    y_pred_train = grid_search.predict(x_val)
    
    # Apply inverse log transformation to get predictions on original scale
    #y_pred_original_scale = inverse_log_transform(y_pred_log)
    
    # Calculate mean squared error on original scale
    mse_original_scale = mean_squared_error(y_val, y_pred_train)
    
    # Calculate mean absolute error on original scale
    mae_original_scale = mean_absolute_error(y_val, y_pred_train)
    
    # Calculate R-squared score on original scale
    r2_original_scale = r2_score(y_val, y_pred_train)
    
    # Print results
    print(f"Model: {model_name}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Mean Squared Error: {grid_search.best_score_}")
    print(f"Test Mean Squared Error: {mse_original_scale}")
    print(f"Test Mean Absolute Error: {mae_original_scale}")
    print(f"Test R-squared: {r2_original_scale}")
    print()


# In[49]:


#Taking Random forest Regressor model as it giving best accuracy
xg=XGBRegressor(learning_rate= 0.1, max_depth= 5, n_estimators= 300)
xg.fit(x_train,y_train)

# Define a function for inverse log transformation
#def inverse_log_transform(y_log):
    #return np.exp(y_log)

# Apply inverse log transformation to target variables
#y_train_original_scale = inverse_log_transform(y_train)
#y_val_original_scale = inverse_log_transform(y_val)

# Predict on validation set
y_pred_train = xg.predict(x_val)

# Apply inverse log transformation to get predictions on original scale
#y_pred_original_scale = inverse_log_transform(y_pred_log)

# Calculate mean squared error on original scale
mse_original_scale = mean_squared_error(y_val, y_pred_train)

# Calculate mean absolute error on original scale
mae_original_scale = mean_absolute_error(y_val, y_pred_train)

# Calculate R-squared score on original scale
r2_original_scale = r2_score(y_val, y_pred_train)

print(f"Mean Squared Error: {mse_original_scale}")
print(f"Mean Absolute Error: {mae_original_scale}")
print(f"R-squared: {r2_original_scale}")
print()


# In[52]:


x_val.shape


# In[53]:


rent_test.shape


# In[54]:


y_pred_test=xg.predict(rent_test)


# In[55]:


y_pred_test=pd.DataFrame(y_pred_test)


# In[56]:


y_pred_test.to_excel('Predicted Rent.xlsx')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




