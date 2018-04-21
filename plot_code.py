# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 14:52:11 2018

@author: Jesus
"""

# Read dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def handle_missing(dataset):
    dataset.project_essay_4.fillna(value="missing", inplace=True)
    dataset.project_essay_3.fillna(value="missing", inplace=True)
    return dataset

df = pd.read_csv('train.csv')
df = handle_missing(df)

y = np.array(list(df['project_is_approved']))
X = list([""] * y.shape[0])

sns.barplot(x=['Disapproved', 'Approved'], y=(np.bincount(y)/y.shape[0]))
plt.title("Project Approved Distribution")
plt.yticks(np.linspace(0,1,11))
plt.show()
print((np.bincount(y)/y.shape[0]))