# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:18:46 2017

@author: vishnu.sk
"""

import pandas as pd
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.kernel_ridge import KernelRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score, mean_absolute_error
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/vishnu.sk/Desktop/LifeCycleSavings.csv')
target="sr"
columns = data.columns.tolist()
columns.remove('sr')
columns.remove('country')
train = data.sample(frac=0.7, random_state=0)
test = data.loc[~data.index.isin(train.index)]
regressor = [
    SVR(kernel='rbf', gamma=0.7, C=1),
    linear_model.Ridge (alpha = .5),
    linear_model.Lasso(alpha = 0.1),
    linear_model.LassoLars(alpha=.1),
    linear_model.BayesianRidge(),
    MLPRegressor(),
    DecisionTreeRegressor(),
    KernelRidge(),
    PassiveAggressiveRegressor(),
    RANSACRegressor(),
    TheilSenRegressor(),
    ]

result_cols = ["Regressor", "Accuracy"]
result_frame = pd.DataFrame(columns=result_cols)

for model in regressor:
    name = model.__class__.__name__
    model.fit(train[columns], train[target])
    result = model.predict(test[columns])
    error = sqrt(mean_squared_error(test[target], result))
    acc = 100 - error
    print (name+' accuracy = '+str(acc)+'%')
    acc_field = pd.DataFrame([[name, acc]], columns=result_cols)
    result_frame = result_frame.append(acc_field)

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Regressor', data=result_frame, color="r")

plt.xlabel('Accuracy %')
plt.title('Regressor Accuracy')
plt.show()  

