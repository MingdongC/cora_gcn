import pandas as pd
import numpy as np
from collections import namedtuple
import torch
import scipy.sparse as sp

def Cora_Data_processing():
    ############导入数据，分隔符为空格##############
    row_data_content = pd.read_csv('./cora/cora.content', sep='\t', header=None)
    print("content shape: ", row_data_content.shape)

    ###########将论文编号转成[0,2707]###############
    map = dict(zip(list(row_data_content[0]), list(row_data_content.index)))

    ###########提取词向量作为特征矩阵################
    features = row_data_content.iloc[ : , 1:-1]
    features = np.array(features)

    ###########对论文类型标签进行one-hot编码#########
    lable = pd.get_dummies(row_data_content[1434])
    lable = np.array(lable)

    ###########导入cite文件，建立邻接矩阵############
    row_data_cites = pd.read_csv('./cora/cora.cites', sep='\t', header=None)
    matrix = np.zeros((lable.shape[0], lable.shape[0]))
    print("cites' shape: ", row_data_cites.shape)
    for i,j in zip(row_data_cites[0], row_data_cites[1]):
        x = map[i]
        y = map[j]
        matrix[x][y] = matrix[y][x] = 1
    adjacency = matrix

    ##########train_mask、val_mask和test_mask用来划分训练集、验证集和测试集###########
    num_nodes = features.shape[0]
    train_index = np.arange(1708)
    val_index = np.arange(1708,2208)
    test_index = np.arange(2208,2708)

    train_mask = np.zeros(num_nodes, dtype=np.bool)
    val_mask = np.zeros(num_nodes, dtype=np.bool)
    test_mask = np.zeros(num_nodes, dtype=np.bool)

    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_index] = True

    ###########Data保存将要return的x，y，adjacency，train_mask，val_mask，test_mask###
    Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])

    ##########打印一下x，y，adjacency，train_mask，val_mask，test_mask的维度###########
    print("node's features shape: ", features.shape)
    print("node's lable shape: ", lable.shape)
    print("node's adjacency shape: ", adjacency.shape)
    print("number of train nodes: ", train_mask.sum())
    print("number of validation nodes: ", val_mask.sum())
    print("number of test nodes: ", test_mask.sum())

    dataset = Data(features, lable, adjacency, train_mask, val_mask, test_mask)
    #Data(x=features, y=lable, adjacency=adjacency, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return dataset
# 规范化邻接矩阵
def normalization(adjacency):
    """
    计算 L=D^-0.5 * (A+I) * D^-0.5
    """
    adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()
# 参数设置
device = "cuda" if torch.cuda.is_available() else "cpu"


# 加载数据，并转换为torch.Tensor
print('=' * 20)
print('加载数据')
dataset = Cora_Data_processing()
x = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
tensor_x = torch.from_numpy(x).to(device)
tensor_y = torch.from_numpy(dataset.y).to(device)
tensor_train_mask = torch.from_numpy(dataset.train_mask).to(device)
tensor_val_mask = torch.from_numpy(dataset.val_mask).to(device)
tensor_test_mask = torch.from_numpy(dataset.test_mask).to(device)
normalization_adjacency = normalization(dataset.adjacency)  # 规范化邻接矩阵
indices = torch.from_numpy(
    np.asarray([normalization_adjacency.row, normalization_adjacency.col]).astype('int64')).long()
values = torch.from_numpy(normalization_adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values, (2708, 2708)).to(device)
