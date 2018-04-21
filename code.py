# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 08:53:48 2018

@author: Jesus
"""
import pandas as pd
import numpy as np
from utils import custom_tokenizer

def handle_missing(dataset):
    dataset.project_essay_4.fillna(value="missing", inplace=True)
    dataset.project_essay_3.fillna(value="missing", inplace=True)
    return dataset

# Read dataset
df = pd.read_csv('train.csv')
df = handle_missing(df)

y = np.array(list(df['project_is_approved']))
X = list([""] * y.shape[0])

# Select Certain Features that are assumed to be more informative for the prediction
selected_feature = ['project_title','project_resource_summary']

# Concatinating text data
for sf in selected_feature:
    X = ["{} {}".format(a_, b_) for a_, b_ in zip(X, df[sf])]

X = np.array(X)

y = y[:100]
X = X[:100]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.80,test_size=0.20,random_state=0)

# Bag of words representation 
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=10, tokenizer=custom_tokenizer).fit(X_train)

## Create bag-of-words representation for the training data and test data
#X_train_bag = vect.transform(X_train)
#X_test_bag = vect.transform(X_test)
#print("train: bag_of_words: {}".format(X_train_bag.shape))
#print("test: bag_of_words: {}".format(X_test_bag.shape))
#
##from sklearn.preprocessing import StandardScaler
##ppr = StandardScaler()
##
##ppr.fit(X_train_bag)
##X_train_bag = ppr.transform(X_train_bag)
##X_test_bag = ppr.transform(X_test_bag)
#
## Prints length of the vocabulary 
#print("Vocabulary size: {}".format(len(vect.vocabulary_)))
## Get the feature names from the vectorizer
#vocabulary = np.array(vect.get_feature_names())
#
##regressor = MLPRegressor((100,50))
##regressor.fit(X_train_bag, y_train)
##y_pred = regressor.predict(X_test_bag)
#
## 2. Regression Trees
#from sklearn.tree import DecisionTreeRegressor
#regressor = DecisionTreeRegressor(random_state=83)
#regressor.fit(X_train_bag, y_train)
#y_pred = regressor.predict(X_test_bag)
#
#
#from sklearn.metrics import roc_auc_score
#score = roc_auc_score(y_test,y_pred)
#print(score)

