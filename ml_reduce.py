  
import pandas as pd
import numpy as np
import pickle

#Load the Dataset
df = pd.read_csv("Revised Refined.csv")
# Remove column name 'A' 
df.drop(['Unnamed: 0'], axis = 1 ,inplace = True)
#Select the X and Y Parameters

X = df[[ 'floor_area_sqm', 'Lease_Period', '0 - 300 metres',
       '300 - 600 metres', '600 - 1000 metres', 'Greater than 1000 metres',
       'storey_range_label', 'town_label', 'flat_type_label',
       'flat_model_label']]

y = df[['resale_price']]

#Select the train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Random Forest Regression to the dataset 
# import the regressor 
from sklearn.ensemble import RandomForestRegressor 

 # create regressor object 
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0) 

sv = regressor.fit(X_train,y_train)

#https://stackoverflow.com/questions/43591621/trained-machine-learning-model-is-too-big
from sklearn.externals import joblib
joblib.dump(sv,  'randomforest.pkl',compress=9)

