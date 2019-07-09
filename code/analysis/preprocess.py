import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from multiprocessing import Process, Pipe
import time

train_df = pd.read_csv('data/train.csv')
structure_df = pd.read_csv('data/structures.csv')

y = train_df.values[:,-1]
jobs = 10
splitnum = len(train_df)//jobs+1

# function for multiprocessing
def calc_distance(index, send):
    data = train_df.values[index * splitnum : (index+1) * splitnum]
    molecule = None
    X = np.zeros((len(data),1))
    
    if index == 0:
        for i,row in tqdm(enumerate(data),total=len(data)):
            if row[1] != molecule:
                molecule = row[1]
                molecule_data = structure_df[structure_df['molecule_name'] == molecule]
            X[i,0] = np.linalg.norm(molecule_data.values[row[2],3:6]-molecule_data.values[row[3],3:6])
    else:
        for i,row in enumerate(data):
            if row[1] != molecule:
                molecule = row[1]
                molecule_data = structure_df[structure_df['molecule_name'] == molecule]
            X[i,0] = np.linalg.norm(molecule_data.values[row[2],3:6]-molecule_data.values[row[3],3:6])
    
    send.send(X)
    send.close()

# start multiprocess
start = time.time()
p_jobs = []
pipes = []
for i in range(jobs):
    get_rev, send_rev = Pipe()
    p = Process(target=calc_distance, args=(i,send_rev))
    pipes.append(get_rev)
    p_jobs.append(p)
    p.start()

result = [x.recv() for x in pipes]

for p in p_jobs:
    p.join()

print("process time:{0:.1f}".format(time.time()-start))

result = np.concatenate(result)
pd.to_pickle(result, 'result/train_X.pkl')