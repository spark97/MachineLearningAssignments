# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 18:04:20 2017

@author: spark
"""

import pandas as pd

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
combine = [train_df,test_df] 

print (train_df.info())
print (test_df.info())
null_train = train_df.isnull().sum().sort_values(ascending=False)
print (null_train[null_train>0])
null_test = test_df.isnull().sum().sort_values(ascending=False)
print(null_test[null_test>0])


print ("Before Dropping")
print (train_df.shape)
print (test_df.shape)

#Drop PoolQC, MiscFeature, Alley, Fence from test and train because many are NULL
train_df = train_df.drop(['PoolQC','MiscFeature','Alley','Fence'],axis=1)
test_df = test_df.drop(['PoolQC','MiscFeature','Alley','Fence'],axis=1)

print ("After Dropping")
print (train_df.shape)
print (test_df.shape)
combine = [train_df,test_df]

print (train_df.info())
print (test_df.info())
null_train = train_df.isnull().sum().sort_values(ascending=False)
print (null_train[null_train>0])
null_test = test_df.isnull().sum().sort_values(ascending=False)
print(null_test[null_test>0])
