import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(df_train):
    X = df_train.drop(['label'], axis=1)
    Y = df_train['label']
        # Split training data to 80/20
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80, random_state=42)
    Y_train = Y_train.values.reshape(-1, 1)
    Y_test = Y_test.values.reshape(-1, 1)

    return X_train, Y_train, X_test, Y_test

def encode_labels(Y_train, Y_test):
    train = []
    test = []
    for i in range(len(Y_train)):
        tmp = np.zeros(10)
        tmp[Y_train[i]] = 1
        train.append(tmp)
    for i in range(len(Y_test)):
        tmp = np.zeros(10)
        tmp[Y_test[i]] = 1
        test.append(tmp)
    Y_train = pd.DataFrame(train).values.reshape(-1, 10)
    Y_test = pd.DataFrame(test).values.reshape(-1, 10)
    return Y_train, Y_test
