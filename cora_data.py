import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plot
from collections import namedtuple
import os
import os.path as osp
import pickle
import urllib
import numpy as np
import itertools

#定义一个namedtuple元组来存储下载的数据
Data = namedtuple('Data',['x','y','adjacency','train_mask','val_mask','test_mask'])

class Cora_Data():

    download_url = "https://github.com/kimiyoung/planetoid/tree/master/data"

    #filenames 是一个list，用来储存下载的文件名
    filenames = ["ind.cora.{}".format(name) for name in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="cora", rebuild=False):
        """
        包括数据下载、处理、加载等功能
        当数据的缓存文件存在时，将使用缓存文件，否则将下载、处理，并缓存到磁盘

        :param data_root: string， optional
            存放数据的目录，原始数据路径：{data_root}/raw
            缓存数据路径： {data_root}/processed_cora.pkl
        :param rebuild:  boolean, optional
            是否需要重新构建数据集，当设为True时，如果缓存数据存在也会重建数据
        """

        self.data_root = data_root

        #join 是将data_root目录和”processed_cora.pkl“合成一个路径
        save_file = osp.join(self.data_root, "processed_cora.pkl")

        #exists函数是如果save_file的路径存在则返回true 否则返回false
        #data前面加”_“表示不能用”_data"不能用from  import 导入
        #如果save_file路径存在并且不需要重构数据，则使用缓存数据，即存在save_file里的路径
        if osp.exists(save_file) and not rebuild :
            print("Using Cached file: {}".format(save_file))

            # pickle的作用是将文件序列化。
            self._data = pickle.load(open(save_file, "rb"))

        #如果save_file路径不存在或是rebuild等于true，则执行
        else :
            self.maybe_download()
            self._data = self.processing_data()

            #pickle.dump是将self.data的数据写入f中
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)

            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        #返回Data数据对象，包括x，y，adjacency，train_mask, val_mask, test_mask

        return self._data

    def maybe_download(self):

        save_path = os.path.join(self.data_root, "raw")
        for name in self.filenames:
            if not osp.exists(osp.join(save_path, name)):
                self.download_data("{}/{}".format(self.download_url, name), save_path)

    @staticmethod
    def download_data(url, save_path):
        #数据下载工具，当原始数据不存在时将会进行下载

        if not os.path.exists(save_path):

            #makedirs 创建save_path的文件夹
            os.makedirs(save_path)

        #urllib是pythen内置的http请求库，request是网页请求
        for i in range(0,100):

            try:
                data = urllib.request.urlopen(url, timeout=10)
                break
            except Exception as e:
                print("出现异常：", str(e))

        #splitext 分割路径，返回路径名和扩展名的元组
        filename = osp.basename(url)

        with open(os.path.join(save_path, filename), 'wb') as f:
            for i in range(0,100):
                try:
                    f.write(data.read())
                    break
                except Exception as e:
                    print("出现异常：", str(e))

        return True

    def processing_data(self):
        #处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集

        print("processing data......")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(osp.join(self.data_root, "raw", name)) for name in self.filenames]
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx,tx) ,axis=0)
        y = np.concatenate((ally,ty), axis=0).argmax(axis=0)

        x[test_index] = [sorted_test_index]
        y[test_index] = [sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Lable's feature shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of valitation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x,y=y,adjacency=adjacency,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        #根据邻接表创建邻接矩阵

        edge_index = []
        num_nodes = len(adj_dict)
        for src,dst in adj_dict.item():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v,src] for v in dst)

        #上述得到的结果存在重复边，以下删掉重复边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])), shape=(num_nodes, num_nodes), dtype="float32")

        return adjacency

    @staticmethod
    def read_data(path):
        #使用不同的方式读取原数据以进一步处理

        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out

        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            print("111")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

































































































