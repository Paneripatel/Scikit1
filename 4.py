'''
4 Problem 4 : How to Get Regression Model Summary from Scikit-Learn (https://www.geeksforgeeks.org/how-to-get-regression-model-summary-from-scikit-learn/)
'''

# Import packages
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.datasets import load_iris # type: ignore

# Load the data
irisData = load_iris()

# Create feature and target arrays
X = irisData.data
y = irisData.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

# predicting on the X_test data set
print(model.predict(X_test))

# summary of the model
print('model intercept :', model.intercept_)
print('model coefficients : ', model.coef_)
print('Model score : ', model.score(X, y))