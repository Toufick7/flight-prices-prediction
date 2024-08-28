#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Cufflinks is another library that connects the Pandas data frame with Plotly enabling users 
to create visualizations directly from Pandas. The library binds the power of Plotly with the 
flexibility of Pandas for easy plotting


The plotly Python library is an interactive, open-source plotting library that supports 
over 40 unique chart types covering a wide range of statistical, financial, geographic, 
scientific, and 3-dimensional use-cases.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chart_studio.plotly import plot,iplot
import cufflinks as cf
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[2]:


os.getcwd()


# In[3]:


os.chdir ('C:\\Noble\\Training\\Top Mentor\\Training\Presentation\\Project\\Project 12 Flight Price Predict Heroku\\')
os.getcwd()


# In[4]:


df=pd.read_excel("Data_Train.xlsx")
display(df)


# In[5]:


#Automated Exploratory Data Analysis (EDA) 
#Pandas Profiling Report 

import pandas_profiling as pf
display(pf.ProfileReport(df))


# In[6]:


# Number of records
len(df)


# In[7]:


# Number of records
display (df.shape)


# In[8]:


#checking the data types
display (df.dtypes )


# In[9]:


#Checking null values
display (df.isna().sum() )


# In[10]:


#Remove the NaN values (records) from the dataset
df.dropna(how='any',inplace=True)
df.isnull().sum()


# In[11]:


# Display Number of records- Number of records reduced by 1 , since we removed null record
display (df.shape)


# In[12]:


df.head()


# In[13]:


# Feature Engineering
#Split Date of Journey column to Day and Month 
# Convert Date_of_Journey to its appropriate format as datetime with regards to day and month. Added two additional columns Day and Month
df['Date_of_Journey']=pd.to_datetime(df['Date_of_Journey']) # Chanage Data type- from Object to Date and Time
df['Day_of_Journey']=(df['Date_of_Journey']).dt.day # Day column 
df['Month_of_Journey']=(df['Date_of_Journey']).dt.month # Month Column 
display(df)


# In[14]:


# Drop the column - Date_of_journey
df.drop(["Date_of_Journey"],axis=1,inplace=True)
display(df.head())


# In[15]:


#convert to datetime and Split Dep_Time column to hour and minutes 
df['Dep_hr']=pd.to_datetime(df['Dep_Time']).dt.hour
df['Dep_min']=pd.to_datetime(df['Dep_Time']).dt.minute
display(df.head())


# In[16]:


#Drop the column 'Dep_Time'

df.drop(["Dep_Time"],axis=1,inplace=True)
display(df.head())


# In[17]:


#convert to datetime and Split Arrival_Time column to hour and minutes 
df['Arrival_hr']=pd.to_datetime(df['Arrival_Time']).dt.hour
df['Arrival_min']=pd.to_datetime(df['Arrival_Time']).dt.minute
display(df.head())


# In[18]:


#Drop the column 'Arrival_Time'

df.drop(["Arrival_Time"],axis=1,inplace=True)
display(df.head())


# In[19]:



display (df['Duration'])


# In[20]:


'''
split duration datapoints based on space ' '

expand : bool, default False

Expand the splitted strings into separate columns.

If True, return DataFrame/MultiIndex expanding dimensionality.
If False, return Series/Index, containing lists of strings.
'''

duration=df['Duration'].str.split(' ',expand=True) 
display (duration)


# In[21]:


#In column 1 ie minutes column fill all NULL values with  '00m'
duration[1].fillna('00m',inplace=True)  
display (duration)


# In[22]:


#Extract the hours ie 0th column by excluding last character h x[:-1]
#select the item at index o and leave the last one (in this case the 'h')
df['duration_hr']=duration[0].apply(lambda x: x[:-1]) 
display (df['duration_hr'])


# In[23]:


#Extract the minutes, select the item at index 1 and leave the last one (in this case the 'm')
df['duration_min']=duration[1].apply(lambda x: x[:-1]) 
display (df['duration_min'])


# In[24]:


#Drop the column'Duration'

df.drop(["Duration"],axis=1,inplace=True)
display (df)


# In[25]:


# Config file
cf.set_config_file(theme='ggplot',sharing='public',offline=True)


# In[26]:


# Count of Airlines 
df['Airline'].value_counts()


# In[27]:


#Airline VS average Price
#Jet Airways Business has the highest price with Trujet having the lowest
Airprices=df.groupby('Airline')['Price'].mean().sort_values(ascending=False)
plt.figure(figsize=(15,10))
sns.barplot(Airprices.index,Airprices.values)
plt.xticks(rotation=270)
plt.show()


# In[28]:


# Box Plot Airline VS Price
plt.figure(figsize=(20,10))
sns.boxplot(y='Price',x='Airline',data= df.sort_values('Price',ascending=False))
plt.show()


# In[29]:


# Price based on number of stops
df.groupby(['Airline','Total_Stops'])['Price'].mean()


# In[30]:


# Bar Plot - Same Details as chart 
#One stop and two stops Jet Airways Business is having the highest price

plt.figure(figsize=(18,10))
ax=sns.barplot(x=df['Airline'],y=df['Price'],hue=df['Total_Stops'],palette="Set1")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()


# In[31]:


# Number of flights from different Airports
df['Source'].value_counts()


# In[32]:


# Source vs Price
plt.figure(figsize=(15,10))
sns.barplot(y='Price',x='Source',data=df.sort_values('Price',ascending=False))
plt.show()


# In[33]:


# Flights in the destination
df['Destination'].value_counts()


# In[34]:


#Destination vs Price
plt.figure(figsize=(15,10))
sns.barplot(y='Price',x='Destination',data=df.sort_values('Price',ascending=False))
plt.show()


# In[35]:


# There is New Delhi and Delhi in the data set, replace New Delhi with Delhi
for i in df:
    df.replace('New Delhi','Delhi',inplace=True)


# In[36]:


# Display Unique Destinations
display(df['Destination'].unique())


# In[37]:


df['Destination'].value_counts()


# In[38]:


# Create Bar Plot again with Price 
plt.figure(figsize=(15,10))
sns.barplot(y='Price',x='Destination',data=df.sort_values('Price',ascending=False))
plt.show()


# In[39]:


# Create Heat Map
#The features are less correlated which is a good thing for us to avoid Multicollinearity
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[40]:


display (df.head(4))


# In[41]:


# Label Encoding - Column 'Total_Stops'
df['Total_Stops']=df['Total_Stops'].map({'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4})
display (df.head(4))


# In[42]:


# One Hot Encoding -Column "Airline"- Due to drop_first, there is no column for Air Asia
air_dummy=pd.get_dummies(df['Airline'],drop_first=True)
display (air_dummy)


# In[43]:


#One hot encoding for columns 'Source','Destination' Due to drop first Source Cochin column Dropped 
source_dest_dummy=pd.get_dummies(df[['Source','Destination']],drop_first=True)
display (source_dest_dummy)


# In[44]:


# Concatenate Label Encoded and one hot encoded columns into main data frame
df=pd.concat([air_dummy,source_dest_dummy,df],axis=1)
display (df.head(4))


# In[45]:


#Drop Columns -Already Label Encoded or One Hot Encoded 
df.drop(['Airline','Source','Destination'],inplace=True,axis=1)


# In[46]:


display (df.shape)


# In[47]:


# Read Test Data
df_test=pd.read_excel("Test_set.xlsx")
display(df_test)


# In[48]:


df_test['Date_of_Journey']=pd.to_datetime(df_test['Date_of_Journey'])
df_test['Day_of_Journey']=(df_test['Date_of_Journey']).dt.day
df_test['Month_of_Journey']=(df_test['Date_of_Journey']).dt.month

#Dep_time 
df_test['Dep_hr']=pd.to_datetime(df_test['Dep_Time']).dt.hour
df_test['Dep_min']=pd.to_datetime(df_test['Dep_Time']).dt.minute

#Arrival_time
df_test['Arrival_hr']=pd.to_datetime(df_test['Arrival_Time']).dt.hour
df_test['Arrival_min']=pd.to_datetime(df_test['Arrival_Time']).dt.minute

#Splitting duration  time

a=df_test['Duration'].str.split(' ',expand=True)
a[1].fillna('00m',inplace=True)
df_test['dur_hr']=a[0].apply(lambda x: x[:-1])
df_test['dur_min']=a[1].apply(lambda x: x[:-1])

#dropping the data
df_test.drop(['Date_of_Journey','Duration','Arrival_Time','Dep_Time'],inplace=True,axis=1)

#Handling Categorical Values 
df_test['Total_Stops']=df_test['Total_Stops'].map({'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4})

air_dummy=pd.get_dummies(df_test['Airline'],drop_first=True)
source_dest_dummy=pd.get_dummies(df_test[['Source','Destination']],drop_first=True)
df_test=pd.concat([air_dummy,source_dest_dummy,df_test],axis=1)


# In[49]:


# Drop additional Columns
df_test.drop(['Airline','Source','Destination','Additional_Info',"Route"],inplace=True,axis=1)
display (df_test.head(4))


# In[50]:


print('train_shape',df.shape)
# Additional columns in training data set  'Route', 'Price','Additional_Info', can be removed later 
print('test_shape',df_test.shape)


# In[51]:


# Create X and Y from Training Data 

x=df.drop(['Route', 'Price','Additional_Info'],axis=1)
y=df['Price']


# In[52]:


display (x.head(3))


# In[53]:


#ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
ET_Model=ExtraTreesRegressor()
ET_Model.fit(x,y)


# In[54]:


# Predict and Print Accuracy 
y_predict=ET_Model.predict(x)
display (r2_score(y,y_predict))


# In[55]:


#Feature Importance Graph
pd.Series(ET_Model.feature_importances_,index=x.columns).sort_values(ascending=False).plot(kind='bar',figsize=(18,10))


# In[56]:


#splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 50)


# In[57]:


#Preparing Extra Tree Regression with Training Data 
from sklearn.ensemble import  ExtraTreesRegressor
ET_Model=ExtraTreesRegressor(n_estimators = 120)
ET_Model.fit(X_train,y_train)


# In[58]:


# Prediction and Print Accuracy
y_predict=ET_Model.predict(X_test)
from sklearn.metrics import r2_score
display (r2_score(y_test,y_predict))


# In[59]:


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
RF_Model=RandomForestRegressor()
RF_Model.fit(X_train,y_train)
y_predict=RF_Model.predict(X_test)
r2_score(y_test,y_predict)


# In[60]:


# Hyperparameter Tuning and RandomizedSearchCV - Model used - RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 80, stop = 1500, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(6, 45, num = 5)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

# create random grid

rand_grid={'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf=RandomForestRegressor()

rCV=RandomizedSearchCV(estimator=rf,param_distributions=rand_grid,scoring='neg_mean_squared_error',n_iter=10,cv=3,random_state=42, n_jobs = 1)


# In[61]:


# Fit Model
rCV.fit(X_train,y_train)


# In[62]:


# Prediction 
rf_pred=rCV.predict(X_test)
display (rf_pred)


# In[63]:


# mean_absolute_error and mean_squared_error
from sklearn.metrics import mean_absolute_error,mean_squared_error
print('MAE',mean_absolute_error(y_test,rf_pred))
print('MSE',mean_squared_error(y_test,rf_pred))


# In[64]:


# Display Accuracy
display (r2_score(y_test,rf_pred))


# In[65]:


# Model CatBoostRegressor
from catboost import CatBoostRegressor
cat=CatBoostRegressor()
cat.fit(X_train,y_train)


# In[66]:


# Cat Boost Prediction 
cat_pred=cat.predict(X_test)
display (cat_pred)


# In[67]:


# Cat Boost Accuracy
display (r2_score(y_test,cat_pred))


# In[68]:


# Change the data type for Light GBM Regressor - Convert to Integer 
X_train[['duration_hr','duration_min']]=X_train[['duration_hr','duration_min']].astype(int)
X_test[['duration_hr','duration_min']]=X_test[['duration_hr','duration_min']].astype(int)


# In[69]:


# Create Model LGBMRegressor
from lightgbm import LGBMRegressor

lgb_model = LGBMRegressor()
lgb_model.fit(X_train,y_train)


# In[70]:


# Prediction and display accuracy 
lgb_pred=lgb_model.predict(X_test)
display (r2_score(y_test,lgb_pred))


# In[71]:


# Create Model XG Boost Regressor 
import xgboost as xgb
xgb_model=xgb.XGBRegressor()
xgb_model.fit(X_train,y_train)
xgb_pred=xgb_model.predict(X_test)
display (r2_score(y_test,xgb_pred))


# In[72]:


# Display top 5 records
df.head()


# In[73]:


# #Use pickle to save our model so that we can use it later

import pickle 
# Saving model to disk
pickle.dump(cat, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


# In[74]:


# Display column names
display (df.columns)


# In[75]:


# Create the data set for deployment by removing columns Route and Additional_Info
deploy_df=df.drop(['Route','Additional_Info'],axis=1)


# In[77]:


# Generate the .csv file and display the data set 
deploy_df.to_csv('deploy_df.csv')
display (deploy_df)


# In[ ]:




