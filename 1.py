'''
Scikit1

1 Problem 1 : Save and Load Machine Learning Models in Python with scikit-learn (https://www.geeksforgeeks.org/save-and-load-machine-learning-models-in-python-with-scikit-learn/)
'''

# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn import metrics # type: ignore
import pickle

# import the dataset
dataset = pd.read_csv('headbrain1.csv')

X = dataset.iloc[:, : -1].values
Y = dataset.iloc[:, -1].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# create a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# save the model
filename = 'linear_model.sav'
pickle.dump(regressor, open(filename, 'wb'))

# load the model
load_model = pickle.load(open(filename, 'rb'))

y_pred = load_model.predict(X_test)
print('root mean squared error : ', np.sqrt(
    metrics.mean_squared_error(y_test, y_pred)))