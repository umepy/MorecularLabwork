import numpy as np
import time
from feature.atom_distance import atom_distance
from feature_enginner.scalar_coupling_contributions import contribution
from feature_enginner.potential_energy import potential_energy

def merge_features():
    start = time.time()
    train_X = []
    names = []

    # READ each features
    #data, name = atom_distance()
    #train_X.append(data)
    #names.extend(name)

    data, name = contribution()
    train_X.append(data)
    names.extend(name)

    #data, name = potential_energy()
    #train_X.append(data)
    #names.extend(name)

    # FOR CHECKING "TYPE" and "SHAPE" of data
    #for i in train_X:
    #    print(type(i),i.shape)

    train_X = np.hstack(train_X)
    print('Finish Preprocess:{0:.1f} sec'.format(time.time()-start))
    return train_X, names
