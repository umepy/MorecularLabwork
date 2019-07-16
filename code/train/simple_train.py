#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from multiprocessing import Process, Pipe
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import lightgbm as lgb
sys.path.append('../preprocess/')
from main_merge import merge_features
from file_read import get_df
from first_train import train

if len(sys.argv)==1:
    FIRST_TRAIN = True
else:
    FIRST_TRAIN = False

df_list = get_df(['train','structures'])
train_df = df_list[0]
structure_df = df_list[1]

X, names = merge_features()
y = train_df.values[:,-1]

def Ad_split(X,y,names):
    X_df = pd.concat([pd.DataFrame(X, columns=names),
                      pd.DataFrame(y, columns=['target'])],axis=1)
    X_df = X_df.sample(frac=1, random_state=0)
    first_df = X_df[:len(X_df//2)]
    first_df = first_df.drop(['target'],axis=1)
    second_arr = X_df.loc[:len(X_df)-len(first_df)].values
    firstX_column = [n for n in names if 'AD_'!=n[:3]]
    firsty_column = [n for n in names if 'AD_'==n[:3]]
    first_X = first_df[firstX_column].values
    first_y = first_df[firsty_column].values
    first_X = first_X.reshape(len(first_X),len(firstX_column)).astype(np.float32)
    first_y = first_y.reshape(len(first_y),len(firsty_column)).astype(np.float32)
    second_X = second_arr[:,:-1].astype(np.float32)
    second_y = y.astype(np.float32)
    return first_X, first_y, second_X, second_y

def Feature_importance(fti, names):
    dic = {}
    for i in range(len(fti)):
        dic[names[i]] = fti[i]
    for k, v in sorted(dic.items(),key=lambda x :x[1], reverse=True):
        print(str(k) + ": " + str(v))

def Evaluation(y, pred):
    types = train_df['type'].values
    dic = {'1JHC':[], '2JHH':[], '1JHN':[], '2JHN':[], '2JHC':[], '3JHH':[], '3JHC':[], '3JHN':[]}
    loss = np.abs(y - pred)
    score=0
    for i in range(len(y)):
        dic[types[i]].append(loss[i])
    for i in dic.keys():
        score+=np.log(np.sum(dic[i])/len(dic[i]))
    score/=len(dic)
    return score

first_X,first_y, second_X, second_y = Ad_split(X,y,names)

if FIRST_TRAIN:
    train(first_X, first_y, max_epoch=40)

kf = KFold(n_splits=10)
score = []
for train_idx, test_idx in tqdm(kf.split(second_X), total=10):
    train_X, test_X = second_X[train_idx], second_X[test_idx]
    train_y, test_y = second_y[train_idx], second_y[test_idx]

    #model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    model = lgb.LGBMModel(objective='regression')
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    score.append(Evaluation(test_y, pred))

print('10-Fold score:', np.mean(score))
print('Feature Importance')
model.booster_.save_model('second_model.txt')
Feature_importance(model.feature_importances_, names)
