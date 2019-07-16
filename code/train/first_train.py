#!/usr/bin/env python
# coding: utf-8

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.training import extensions
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

class first_dataset(chainer.dataset.DatasetMixin):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def get_example(self,i):
        data = chainer.datasets.TupleDataset(self.X, self.y)
        return data[i]

class NN(chainer.Chain):
    def __init__(self, n_out):
        super(NN, self).__init__(
            l1 = L.Linear(None, 64),
            l2 = L.Linear(None, 64),
            l3 = L.Linear(None, n_out)
        )
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

def train(X, y, batch_size=256, max_epoch=20, gpu_id=0):
    n_out = y.shape[1]
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75)
    train = first_dataset(train_X, train_y)
    test = first_dataset(test_X, test_y)
    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size, False, False)

    model = NN(n_out)
    if gpu_id>=0:
        model.to_gpu(gpu_id)
    model = L.Classifier(model, lossfun=F.mean_squared_error, accfun=F.r2_score)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    updater = chainer.training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = chainer.training.Trainer(updater, (max_epoch,'epoch'), out='first_result')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(model.predictor, filename='first_model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.DumpGraph('main/loss'))
    trainer.run()
