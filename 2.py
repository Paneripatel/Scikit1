'''
2 Problem 2 : Data Preprocessing, Analysis, and Visualization for building a Machine learning model (https://www.geeksforgeeks.org/data-preprocessing-analysis-and-visualization-for-building-a-machine-learning-model/)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

dataset = pd.read_csv('Churn_Modelling.csv')

dataset.head()

dataset.info()

plt.figure(figsize=(12,6))

sns.heatmap(dataset.corr(),
            cmap='BrBG',
            fmt='.2f',
            linewidths=2,
            annot=True)

lis = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
plt.subplots(figsize=(15, 8))
index = 1

for i in lis:
    plt.subplot(2, 2, index)
    sns.distplot(dataset[i])
    index += 1

lis2 = ['Geography', 'Gender']
plt.subplots(figsize=(10, 5))
index = 1

for col in lis2:
    y = dataset[col].value_counts()
    plt.subplot(1, 2, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1

dataset.isnull().any()

dataset["Geography"].fillna(dataset["Geography"].mode()[0],inplace = True)
dataset["Gender"].fillna(dataset["Gender"].mode()[0],inplace = True)
dataset["Age"].fillna(dataset["Age"].mean(),inplace = True)

dataset.isnull().any()

le = LabelEncoder()
dataset['Geography'] = le.fit_transform(dataset["Geography"])
dataset['Gender'] = le.fit_transform(dataset["Gender"])

x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13:14].values

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size = 0.2, 
                                                    random_state = 0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore

from sklearn import metrics # type: ignore

knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators = 7,
                             criterion = 'entropy',
                             random_state =7)
svc = SVC()
lc = LogisticRegression()

# making predictions on the training set
for clf in (rfc, knn, svc,lc):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy score of ",clf.__class__.__name__,"=",
          100*metrics.accuracy_score(y_test, y_pred))
    
    