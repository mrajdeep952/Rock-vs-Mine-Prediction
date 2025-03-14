# Importing the dependencies
     

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
     

# Data Collection and Data Processing
     

# Loading the Dataset to a pandas Dataframe
     

sonar_data_df = pd.read_csv('/content/sonar.csv', header=None)
     

sonar_data_df.head()

# number of rows and columns or shape of the dataframe
sonar_data_df.shape
     

# Getting the statistical measures for the donar dataset
sonar_data_df.describe()

sonar_data_df[60].value_counts()

# M ---> Mine
# R ---> Rock
     

sonar_data_df.groupby(60).mean()

# Separating the data and lables
X = sonar_data_df.drop(columns=60, axis=1)
Y = sonar_data_df[60]
     

print(X)
print(Y)

# Dividing the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)
     

print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

# using the Logistic Regression model
model = LogisticRegression()
     

# Training the Logistic Regression model with training data
model.fit(X_train, Y_train)
     
# Model evaluation
     

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
     

# Accuracy score on training data
print(training_data_accuracy)

# Accuracy on testing data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
     

print(testing_data_accuracy)

# Making the preddictive System

input_data = (0.0366,0.0421,0.0504,0.0250,0.0596,0.0252,0.0958,0.0991,0.1419,0.1847,0.2222,0.2648,0.2508,0.2291,0.1555,0.1863,0.2387,0.3345,0.5233,0.6684,0.7766,0.7928,0.7940,0.9129,0.9498,0.9835,1.0000,0.9471,0.8237,0.6252,0.4181,0.3209,0.2658,0.2196,0.1588,0.0561,0.0948,0.1700,0.1215,0.1282,0.0386,0.1329,0.2331,0.2468,0.1960,0.1985,0.1570,0.0921,0.0549,0.0194,0.0166,0.0132,0.0027,0.0022,0.0059,0.0016,0.0025,0.0017,0.0027,0.0027)

# changing the input array into numpy array
input_data_as_npArray =  np.asarray(input_data)

# reshaping the np array as we are predicting for one instance
reshaped_input_data = input_data_as_npArray.reshape(1, -1)

prediction = model.predict(reshaped_input_data)

# print(prediction)

if(prediction[0]  == 'R'):
  print('The object is a Rock')
  
else:
  print('The object is a Mine')
  
  
# Project Finished