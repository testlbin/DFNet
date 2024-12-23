import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

def _load_datasets(patient_name, test_filenumber):

    """
    下面是新改的
    :param patient_name 患者姓名
    :param test_filenumber 样本总数目
    :return:
    """
    # patient_name = '合集分类'
    # test_filenumber = 2064

    # patient_name = '丁以宏'
    # test_filenumber = 445
    # patient_name = '丁以宏'
    # test_filenumber = 445
    # patient_name = '沈羽'
    # test_filenumber = 410
    # patient_name = '周晨斌'
    # test_filenumber = 249
    # patient_name = '洪铮铭'
    # test_filenumber = 183
    # patient_name = '侯润龙'
    # test_filenumber = 159
    # patient_name = '金日浩'
    # test_filenumber = 132
    # patient_name = '孔泽斌'
    # test_filenumber = 262
    # patient_name = '丁渊'
    # test_filenumber = 556
    # patient_name = '丁彤彬'
    # test_filenumber = 68
    # patient_name = '伍思宇'
    # test_filenumber = 67



    y_train = pd.read_csv('Hanshen/五分类数据/Train/Train-label.csv')
    # y_val = pd.read_csv('Hanshen/Val_label.csv')
    # y_test = pd.read_csv('Hanshen/五分类数据/Train/'+ patient_name + '/label.csv')
    y_test = pd.read_csv('Hanshen/五分类数据/Train/' + patient_name + '/Trainlabel.csv')


    # 获得csv 中的数据
    train_label = y_train.label.values
    # val_label = y_val.label.values
    test_label = y_test.label.values

    """
    提取padding前的length 和 多变量数据 存到数组
    """
    # 标准化函数
    min_max_scaler = preprocessing.MinMaxScaler()

    tr_data = []
    # val_data = []
    test_data= []

    for i in range(1, test_filenumber):

        train_data_path = 'Hanshen/五分类数据/Train/' + patient_name + '/Padding/{}.csv'.format(i)
        # train_data_path = 'Hanshen/病人1/Padding/{}.csv'.format(i)
        tr_getdata = pd.read_csv(train_data_path, header=None)

        test_data1 = np.asarray(tr_getdata)
        # tes_data = tes_data.T
        test_data1 = min_max_scaler.fit_transform(test_data1)   # Normalizing

        test_data.append(test_data1)

    for i in range(1, 956):

        test_data_path1 = 'Hanshen/五分类数据/Train/Train/pad/{}.csv'.format(i)
        test_getdata1 = pd.read_csv(test_data_path1, header=None)

        tte_data1 = np.asarray(test_getdata1)
        # t_r_data = t_r_data.T
        # 标准化
        tte_data2 = min_max_scaler.fit_transform(tte_data1)

        tr_data.append(tte_data2)

    # for i in range(1, 222):
    #
    #     test_data_path1 = 'Hanshen/五分类数据/Train/刘腾/Padding/{}.csv'.format(i)
    #     test_getdata1 = pd.read_csv(test_data_path1, header=None)
    #
    #     tte_data1 = np.asarray(test_getdata1)
    #     # t_r_data = t_r_data.T
    #     # 标准化
    #     tte_data2 = min_max_scaler.fit_transform(tte_data1)
    #
    #     tr_data.append(tte_data2)


    # for i in range(1, 401):
    #
    #     test_data_path2 = 'Hanshen/五分类数据/Train/沈1/pad/{}.csv'.format(i)
    #     test_getdata2 = pd.read_csv(test_data_path2, header=None)
    #
    #     tte_data3 = np.asarray(test_getdata2)
    #     # t_r_data = t_r_data.T
    #     # 标准化
    #     tte_data4 = min_max_scaler.fit_transform(tte_data3)
    #
    #     test_data.append(tte_data4)

    # for i in range(1, 173):
    #
    #     test_data_path2 = 'Hanshen/病人2/Padding/{}.csv'.format(i)
    #     test_getdata2 = pd.read_csv(test_data_path2, header=None)
    #
    #     tr2_data = np.asarray(test_getdata2)
    #     # tes_data = tes_data.T
    #
    #     # 标准化
    #     test_data2 = min_max_scaler.fit_transform(tr2_data)
    #
    #     test_data.append(test_data2)
    #
    # for i in range(1, 630):
    #
    #     test_data_path2 = 'Hanshen/病人4/padding/{}.csv'.format(i)
    #     test_getdata2 = pd.read_csv(test_data_path2, header=None)
    #
    #     tr1_data = np.asarray(test_getdata2)
    #     # tes_data = tes_data.T
    #
    #     # 标准化
    #     test_data1 = min_max_scaler.fit_transform(tr1_data)
    #
    #     test_data.append(test_data1)



# 训练集
#
#     for i in range(1, 559):
#
#         train_data_path = 'Hanshen/病人3/padding/{}.csv'.format(i)
#         tr_getdata = pd.read_csv(train_data_path, header=None)
#
#         t_r_data = np.asarray(tr_getdata)
#         # t_r_data = t_r_data.T
#         # 标准化
#         t_r_data = min_max_scaler.fit_transform(t_r_data)
#
#         tr_data.append(t_r_data)
#
#     for i in range(1, 401):
#
#         test_data_path = 'Hanshen/病人1/Padding/{}.csv'.format(i)
#         test_getdata = pd.read_csv(test_data_path, header=None)
#
#         tr2_data = np.asarray(test_getdata)
#         # tes_data = tes_data.T
#
#         # 标准化
#         tr2_data = min_max_scaler.fit_transform(tr2_data)
#
#         tr_data.append(tr2_data)
#
#
# # 测试集
#
#     # for i in range(1, 61):
#     #
#     #     val_data_path = 'Hanshen/Padding/val/{}.csv'.format(i)
#     #     val_getdata = pd.read_csv(val_data_path, header=None)
#     #
#     #     v_data = np.asarray(val_getdata)
#     #     # v_data = v_data.T
#     #
#     #     # 标准化
#     #     # v_data = min_max_scaler.fit_transform(v_data)
#     #
#     #     val_data.append(v_data)
#
#     for i in range(1, 630):
#
#         test_data_path = 'Hanshen/病人4/padding/{}.csv'.format(i)
#         test_getdata = pd.read_csv(test_data_path, header=None)
#
#         tes_data = np.asarray(test_getdata)
#         # tes_data = tes_data.T
#
#         # 标准化
#         tes_data = min_max_scaler.fit_transform(tes_data)
#
#         test_data.append(tes_data)


    # 转换为数组

    tr = np.array(tr_data)
    # val = np.array(val_data)
    tes = np.array(test_data)
    """
    转化为tensor 输入计算
    """
    Y_train = train_label
    # Y_val = val_label
    Y_test = test_label

    X_train = tr
    # X_val = val
    X_test = tes


    return X_train, Y_train, X_test, Y_test


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



