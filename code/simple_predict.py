import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from multiprocessing import Process, Pipe
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import lightgbm as lgb
from main_merge import merge_features


train_df = pd.read_csv('data/train.csv')
structure_df = pd.read_csv('data/structures.csv')

X, names = merge_features()
y = train_df.values[:,-1]

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

kf = KFold(n_splits=10)
score = []
for train_idx, test_idx in tqdm(kf.split(X), total=10):
    train_X, test_X = X[train_idx], X[test_idx]
    train_y, test_y = y[train_idx], y[test_idx]

    #model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    model = lgb.LGBMModel(objective='regression')
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    score.append(Evaluation(test_y, pred))

print('10-Fold score:', np.mean(score))
print('Feature Importance')
Feature_importance(model.feature_importances_, names)