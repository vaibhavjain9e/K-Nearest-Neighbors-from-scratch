import numpy as np
import pandas as pd

def data_processing():
    white = pd.read_csv('heart_disease.csv', low_memory=False, sep=',', na_values='?').values

    [N, d] = white.shape

    np.random.shuffle(white)

    # prepare data
    ntr = int(np.round(N * 0.8))
    nval = int(np.round(N * 0.15))
    ntest = N - ntr - nval
    # spliting training, validation, and test
    Xtrain = np.append([np.ones(ntr)], white[:ntr].T[:-1], axis=0).T
    ytrain = white[:ntr].T[-1].T
    Xval = np.append([np.ones(nval)], white[ntr:ntr + nval].T[:-1], axis=0).T
    yval = white[ntr:ntr + nval].T[-1].T
    Xtest = np.append([np.ones(ntest)], white[-ntest:].T[:-1], axis=0).T
    ytest = white[-ntest:].T[-1].T

    return Xtrain, ytrain, Xval, yval, Xtest, ytest

