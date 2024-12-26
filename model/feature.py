import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

def data_preprocess(path):
    data = np.load(path, allow_pickle=True)
    X = data['data']
    y = data['labels']

    # pad sequences to same length with 0
    X_padded = pad_sequence([torch.tensor(x) for x in X], batch_first=True, padding_value=0)

    # X_padded is now a 3-dimensional tensor with shape (num_samples, max_sequence_length, num_features)

    # standardize the data
    num_samples, max_sequence_length, num_features = X_padded.shape
    X_reshaped = X_padded.reshape(num_samples * max_sequence_length, num_features)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X_reshaped)
    X_standardized = torch.tensor(X_standardized.reshape(num_samples, max_sequence_length, num_features),
                                  dtype=torch.float32)
    X_standardized_array = X_standardized.numpy()

    return X_standardized_array, y

def data_preprocess2(data,labels):
    X = data
    y = labels

    # pad sequences to same length with 0
    X_padded = pad_sequence([torch.tensor(x) for x in X], batch_first=True, padding_value=0)

    # X_padded is now a 3-dimensional tensor with shape (num_samples, max_sequence_length, num_features)

    # standardize the data
    num_samples, max_sequence_length, num_features = X_padded.shape

    X_reshaped = X_padded.reshape(num_samples * max_sequence_length, num_features)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X_reshaped)
    X_standardized = torch.tensor(X_standardized.reshape(num_samples, max_sequence_length, num_features),
                                  dtype=torch.float32)
    X_standardized_array = X_standardized.numpy()

    return X_standardized_array, y

def data_preprocess3(data,labels):
    X = data
    y = labels

    # pad sequences to same length with 0
    X_padded = pad_sequence([torch.tensor(x) for x in X], batch_first=True, padding_value=-1)


    num_samples, max_sequence_length = X_padded.shape

    X_standardized_array = X_padded.numpy()
    X_standardized_array = np.reshape(X_standardized_array, (num_samples, 1, max_sequence_length))
    return X_standardized_array, y

def data_preprocess4(data,labels):
    """
    :param data:
    :param labels:
    :return:
        dont need standardized
    """
    X = data
    y = labels

    # pad sequences to same length with 0
    X_padded = pad_sequence([torch.tensor(x) for x in X], batch_first=True, padding_value=0)


    return X_padded, y


def pad_different_length(dataset1, dataset2):

    # # Load the two datasets
    # dataset1 = np.load('dataset1.npz')['arr_0']
    # dataset2 = np.load('dataset2.npz')['arr_0']

    # Find the maximum number of time steps in both datasets
    max_time_steps = max(dataset1.shape[1], dataset2.shape[1])

    # Find the maximum number of channels in both datasets
    max_channels = max(dataset1.shape[2], dataset2.shape[2])

    # Define the padding sizes for each dimension
    pad_time_steps1 = max_time_steps - dataset1.shape[1]
    pad_time_steps2 = max_time_steps - dataset2.shape[1]
    pad_channels1 = max_channels - dataset1.shape[2]
    pad_channels2 = max_channels - dataset2.shape[2]

    # Pad the datasets to the same shape
    dataset1_padded = np.pad(dataset1, ((0, 0), (0, pad_time_steps1), (0, pad_channels1)))
    dataset2_padded = np.pad(dataset2, ((0, 0), (0, pad_time_steps2), (0, pad_channels2)))

    # Check the new shapes of the padded datasets
    # print(dataset1_padded.shape)
    # print(dataset2_padded.shape)
    return dataset1, dataset2



