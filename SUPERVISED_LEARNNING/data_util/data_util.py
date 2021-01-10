import h5py
import numpy as np
import os
import definitions


def load_sign_dataset():
    train_dataset = h5py.File(
        os.path.join(definitions.ROOT_DIR, "SUPERVISED_LEARNNING\data_util\signs_data\signs_train.h5"), "r")
    train_x = np.array(train_dataset["train_set_x"])
    train_y = np.array(train_dataset["train_set_y"])
    m1 = train_y.shape[0]
    train_y = train_y.reshape(m1, -1)

    test_dataset = h5py.File(
        os.path.join(definitions.ROOT_DIR, "SUPERVISED_LEARNNING\data_util\signs_data\signs_test.h5"), "r")
    test_x = np.array(test_dataset["test_set_x"])
    test_y = np.array(test_dataset["test_set_y"])
    m2 = test_y.shape[0]
    test_y = test_y.reshape(m2, -1)
    return train_x, train_y, test_x, test_y


def flatten(x, y):
    m = x.shape[0]
    x = x.reshape(m, -1).T
    y = y.reshape(m, -1).T
    return x, y


def centralized_x(x):
    x = x / 255
    return x


def one_hot_encoding(y):
    m = y.shape[1]
    one_hot = np.zeros((m, 1))
    for i in range(m):
        if y[0][i] == 3:
            one_hot[i][0] = 1
    return one_hot.T
