# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

#Extracting Independent and dependent Variable  
x_BE= dataset.iloc[:, :-1].values  
y_BE= dataset.iloc[:, 4].values  

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_back = LabelEncoder()
x_BE[:, 3] = labelencoder_back.fit_transform(x_BE[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x_BE = onehotencoder.fit_transform(x_BE).toarray()

# Avoiding the Dummy Variable Trap
x_BE = x_BE[:, 1:]  
  
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_BE, y_BE, test_size= 0.2, random_state=0)  

#Selecting parameters using Backward Regression by keeping P value less than 5%
import statsmodels.formula.api as sm
x_BE=np.append(arr=np.ones((50,1)).astype(int),values= x_BE, axis=1)
x_BE_opt=x_BE[:,[0,1,2,3,4,5]]
regressor_OLS =sm.OLS(endog=y_BE,exog=x_BE_opt).fit()
regressor_OLS.summary()
x_BE_opt=x_BE[:,[0,1,3,4,5]]
regressor_OLS =sm.OLS(endog=y_BE,exog=x_BE_opt).fit()
regressor_OLS.summary()
x_BE_opt=x_BE[:,[0,3,4,5]]
regressor_OLS =sm.OLS(endog=y_BE,exog=x_BE_opt).fit()
regressor_OLS.summary()
x_BE_opt=x_BE[:,[0,3,5]]
regressor_OLS =sm.OLS(endog=y_BE,exog=x_BE_opt).fit()
regressor_OLS.summary()
x_BE_opt=x_BE[:,[0,3]]
regressor_OLS =sm.OLS(endog=y_BE,exog=x_BE_opt).fit()
regressor_OLS.summary()

  
#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor_back= LinearRegression()  
regressor_back.fit(np.array(x_BE_train[:,2]).reshape(-1,1), y_BE_train)  
  
#Predicting the Test set result;  
y_back_pred= regressor_back.predict((x_BE_test[:,2]).reshape(-1,1))  

print('Train Score ackward Regressor: ', regressor_back.score(np.array(x_BE_train[:,2]).reshape(-1,1), y_BE_train))  
print('Test Score  Backward regressor: ', regressor_back.score((x_BE_test[:,2]).reshape(-1,1), y_BE_test))  
