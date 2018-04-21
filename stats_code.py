# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 08:27:05 2018

@author: Jesus
"""
import pandas as pd
import numpy as np

def handle_missing(dataset):
    dataset.project_essay_4.fillna(value="missing", inplace=True)
    dataset.project_essay_3.fillna(value="missing", inplace=True)
    return dataset

df = pd.read_csv('train.csv')
df = handle_missing(df)
df = list(df)

#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.80,test_size=0.20,random_state=0)

y = np.array(df['project_is_approved'])
t = list(df['teacher_id'])

import seaborn as sns
sns.set_style('whitegrid')