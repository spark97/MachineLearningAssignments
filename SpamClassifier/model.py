# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:37:24 2017

@author: spark
"""

import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

#Getting stopwords from nltk 
stop_words = stopwords.words('english')

#Read data fromm CSV file
data = pd.read_csv('spam.csv')

#Modifying the columns
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})

#Adding a new column
data['class'] = data['label'].map({'ham':0,'spam':1})


X = data.drop(["label","class"],axis=1)
Y = data["class"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

