#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from multiprocessing import Process, Pipe
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
#sys.path.append(os.getcwd()+'/code/preprocess/')
sys.path.append('../preprocess/')
print(sys.path)
from main_merge import merge_features
from file_read import get_df

df_list = get_df(['train','structures'])
train_df = df_list[0]
structure_df = df_list[1]

X, names = merge_features()
y = train_df.values[:,-1].astype(np.float32)

def Ad_split(X,y,names,AD=True):
    
    index = np.array(range(0,len(X)),dtype=np.int32)
    np.random.seed(1)
    index = np.random.permutation(index)
    first_idx = index[:len(X)//2]
    second_idx = index[len(X)//2:len(X)]
    normal_idx = np.array([i for i,n in enumerate(names) if n[:3]!='AD_'])
    
    if AD:    
        add_idx = np.array([i for i,n in enumerate(names) if n[:3]=='AD_'])

        first_Xy = X[first_idx,:]
        first_X = first_Xy[:,normal_idx].astype(np.float32)
        first_X = first_X.reshape(len(first_idx),len(normal_idx))
        first_y = first_Xy[:,add_idx].astype(np.float32)
        first_y = first_y.reshape(len(first_idx),len(add_idx))

        second_X = X[second_idx,:]
        second_X = second_X[:,normal_idx].astype(np.float32)
        second_X = second_X.reshape(len(second_idx),len(normal_idx))
        second_y = y[second_idx].astype(np.float32)

        return first_X, first_y, second_X, second_y, second_idx
    else:
        second_X = X[second_idx,:]
        second_X = second_X[:,normal_idx].astype(np.float32)
        second_X = second_X.reshape(len(second_idx),len(normal_idx))
        second_y = y[second_idx].astype(np.float32)
        
        return second_X, second_y, second_idx

def Feature_importance(fti, names):
    dic = {}
    for i in range(len(fti)):
        dic[names[i]] = fti[i]
    for k, v in sorted(dic.items(),key=lambda x :x[1], reverse=True):
        print(str(k) + ": " + str(v))

def Evaluation(y, pred, index):
    type_arr = train_df['type'].values
    types = type_arr[index]
    dic = {'1JHC':[], '2JHH':[], '1JHN':[], '2JHN':[], '2JHC':[], '3JHH':[], '3JHC':[], '3JHN':[]}
    loss = np.abs(y - pred)
    score=0
    for i in range(len(y)):
        dic[types[i]].append(loss[i])
    for i in dic.keys():
        score+=np.log(np.sum(dic[i])/len(dic[i]))
    score/=len(dic)
    return score

def simple_evaluation(y, pred):
    result = np.abs(y-pred)
    result = np.log(result.astype(float))
    result = np.mean(result)
    return result

add_name = np.array([n for n in names if n[:3]=='AD_'])

if len(add_name) != 0:
    first_X, first_y, second_X, second_y, second_index = Ad_split(X,y,names,AD=True)
    
    kf = KFold(n_splits=5)
    mse = []
    for train_idx, test_idx in tqdm(kf.split(first_X), total=5):
        train_X, test_X = first_X[train_idx], first_X[test_idx]
        train_y, test_y = first_y[train_idx], first_y[test_idx]
        vector = []
        for i in range(train_y.shape[1]):
            first_model = lgb.LGBMModel(objective='regression')
            first_model.fit(train_X, train_y[:,i])
            first_model.booster_.save_model('first_model_'+str(i)+'.txt')
            pred = first_model.predict(test_X)
            vector.append(mean_squared_error(test_y[:,i], pred))
        mse.append(vector)
    print('5-Fold score(mse):', np.mean(mse, axis=0))

    add = np.zeros((len(second_X),first_y.shape[1]), dtype=np.float32)
    for i in range(first_y.shape[1]):
        bst = lgb.Booster(model_file='first_model_'+str(i)+'.txt')
        add[:,i] = bst.predict(second_X)
    second_X = np.concatenate([second_X,add],axis=1)

else:
    second_X, second_y, second_index = Ad_split(X,y,names,AD=False)

kf = KFold(n_splits=10)
score = []
for train_idx, test_idx in tqdm(kf.split(second_X), total=10):
    train_X, test_X, shuffle_idx = second_X[train_idx], second_X[test_idx], second_index[test_idx]
    train_y, test_y = second_y[train_idx], second_y[test_idx]

    #model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    model = lgb.LGBMModel(objective='regression')
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    score.append(Evaluation(test_y, pred, shuffle_idx))

print('10-Fold score:', np.mean(score))
print('Feature Importance')
model.booster_.save_model('second_model.txt')
Feature_importance(model.feature_importances_, names)

