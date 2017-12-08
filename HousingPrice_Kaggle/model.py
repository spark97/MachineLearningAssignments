# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 18:04:20 2017

@author: spark
"""
import random as rd
import pandas as pd
import math
from sklearn import linear_model
import numpy as np

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


#Drop PoolQC, MiscFeature, Alley, Fence, FireplaceQu from test and train because many are NULL
train_df = train_df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','Id'],axis=1)
test_df = test_df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','Id'],axis=1)

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



#Filling the missing values for LotFrontage
mean = train_df['LotFrontage'].mean()
std = train_df['LotFrontage'].std()
expected = rd.randrange(math.ceil(mean-std),math.floor(mean+std),1)
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(expected)
test_df['LotFrontage'] = test_df['LotFrontage'].fillna(expected)
combine = [train_df,test_df]


#Filling missing values for Garage
train_df['GarageFinish'] = train_df['GarageFinish'].fillna('NG')
train_df['GarageType'] = train_df['GarageType'].fillna('NG')
train_df['GarageCond'] = train_df['GarageCond'].fillna('NG')
train_df['GarageQual'] = train_df['GarageQual'].fillna('NG')
train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna('0')

test_df['GarageFinish'] = test_df['GarageFinish'].fillna('NG')
test_df['GarageType'] = test_df['GarageType'].fillna('NG')
test_df['GarageCond'] = test_df['GarageCond'].fillna('NG')
test_df['GarageQual'] = test_df['GarageQual'].fillna('NG')
test_df['GarageYrBlt'] = test_df['GarageYrBlt'].fillna('0')

combine = [train_df,test_df]

train_df = train_df.dropna()
test_df = test_df.dropna()



#Convert all string features to numeric

X_train = train_df.drop('Salesprice',axis=1)
Y_train = train_df['SalePrice']
X_test = test_df
regr = linear_model.LinearRegression()
Y_train = Y_train.reshape(Y_train.shape[0], 1)
print (Y_train.shape)
regr.fit(X_train,Y_train)
pred = regr.predict(X_test)
print (pred)

