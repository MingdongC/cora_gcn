import itertools
import os
import os.path as osp
import pickle
import urllib
from collections import namedtuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt

Data = namedtuple('Data', ['x', 'y''adjacency',
                           'train_mask', 'val_mask', 'test_mask'])


class CoraData(object):
    download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="./cora", rebuild=False):
        """Cora数据，包括数据下载，处理，加载等功能
             当数据的缓存文件存在时，将使用缓存文件，否则将下载、进行处理，并缓存到磁盘
             处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
                 * x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
                 * y: 节点的标签，总共包括7个类别，类型为 np.ndarray
                 * adjacency: 邻接矩阵，维度为 2708 * 2708，类型为 scipy.sparse.coo.coo_matrix
                 * train_mask: 训练集掩码向量，维度为 2708，当节点属于训练集时，相应位置为True，否则False
                 * val_mask: 验证集掩码向量，维度为 2708，当节点属于验证集时，相应位置为True，否则False
                 * test_mask: 测试集掩码向量，维度为 2708，当节点属于测试集时，相应位置为True，否则False
             Args:
             -------
                 data_root: string, optional
                     存放数据的目录，原始数据路径: {data_root}/raw
                     缓存数据路径: {data_root}/processed_cora.pkl
                 rebuild: boolean, optional
                     是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据
             """
        self.data_root = data_root
        save_file = osp.join(self.data_root, "processed_cora.pkl")
        if osp.exists(save_file) and not rebuild:
            print(1)
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        # pickle.dump(obj, file, [,protocol])
        # 注解：将对象obj保存到文件file中去。
        # 　　　protocol为序列化使用的协议版本，0：ASCII协议，所序列化的对象使用可打印的ASCII码表示；
        # 　　　1：老式的二进制协议；2：2.3版本引入的新二进制协议，较以前的更高效。其中协议0和1兼容老版本的python。protocol默认值为0。
        # 　　　file：对象保存到的类文件对象。file必须有write()接口， file可以是一个以’w’方式打开的文件或者一个StringIO对象或者其他任何实现write()接口的对象。如果protocol>=1，文件对象需要是二进制模式打开的。
        # pickle.load(file)
        #
        # 注解：从file中读取一个字符串，并将它重构为原来的python对象。
        # 　　 file:类文件对象，有read()和readline()接口。
        else:
            print(2)
            self.maybe_download()  # 下载或使用原始数据集
            self._data = self.process_data()
            with open(save_file, "wb") as f:  # 在with语句块中就可以使用这个变量操作文件。执行with这个结构之后。f会自动关闭。相当于自带了一个finally。,相当于try
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
          处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
          引用自：https://github.com/rusty1s/pytorch_geometric
          Number of training nodes:  140
            Number of validation nodes:  500
            Number of test nodes:  1000
        """

        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(osp.join(self.data_root, "raw", name)) for name in
                                                       self.filenames]
        # test_index就是随机选取的下标,排下序
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)  # 排序

        x = np.concatenate((allx, tx), axis=0)  # 拼接数组
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)  # 每一行的最大值,即哪个类别,返回下表
        # num_nodes 的数量和 x.shape[0] 是一样的,一开始并没有x,x是由tx和 allx 合并的,allx包含 训练集和验证集(1708,1433),tx是测试集包含(1000,1433),y也是,上下两个y不一样的
        # 但实际上 训练集和验证集并没有用完这么多点
        # 这里y重新定义了,定以后的y是(2708,)没有标签

        x[test_index] = x[sorted_test_index]  # 单纯给test_index 的数据排个序
        y[test_index] = y[sorted_test_index]  # 把类别也附上去
        num_nodes = x.shape[0]  # 有多少个点

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)

        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency = self.build_adjacency(graph)
        # ·train_mask、val_mask、test_mask：与节点数相同的掩码，用于划分训练集、验证集、测试集。如代码清单5-1所示：
        # ·adjacency：邻接矩阵，维度为2708×2708，类型为scipy.sparse.coo_matrix；
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        return Data(x=x, y=y, adjacency=adjacency, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

    def maybe_download(self):  # 返回最后文件名,最后的/后面的
        save_path = os.path.join(self.data_root, "raw")
        for name in self.filenames:
            if not osp.exists(osp.join(save_path, name)):
                self.download_data("{}/{}".format(self.download_url, name), save_path)

    @staticmethod
    def build_adjacency(adj_dict):
        """根据邻接表创建邻接矩阵"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # 去除重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))  # 相同的归为一组
        edge_index = np.asarray(edge_index)  # 变成数组
        adjacency = sp.coo_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),  # 相当于(data,(x,y))
                                  shape=(num_nodes, num_nodes), dtype="float32")  # 构造稀疏矩阵
        return adjacency

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            # 将文件内容转为np矩阵内容,用法https://www.cnblogs.com/wuxiaoqian/p/6618617.html
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

    @staticmethod
    def download_data(url, save_path):
        """数据下载工具，当原始数据不存在时将会进行下载"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #urllib是pythen内置的http请求库，request是网页请求
        for i in range(0,100):

            try:
                data = urllib.request.urlopen(url, timeout=10)
                break
            except Exception as e:
                print("出现异常：", str(e))
        filename = os.path.split(url)[-1]

        with open(os.path.join(save_path, filename), 'wb') as f:
            for i in range(0,100):
                try:
                    f.write(data.read())
                    break
                except Exception as e:
                    print("出现异常：", str(e))

        return True

    @staticmethod
    def normalization(adjacency):
        """计算 L=D^-0.5 * (A+I) * D^-0.5"""
        adjacency += sp.eye(adjacency.shape[0])
        degree = np.array(adjacency.sum(1))  # 把列求和
        d_hat = sp.diags(np.power(degree, -0.5).flatten())  # 变一维矩阵
        # matrix([[1, 2, 3],
        #         [4, 5, 6]])
        #     >>> a.flatten()
        #     matrix([[1, 2, 3, 4, 5, 6]])

        return d_hat.dot(adjacency).dot(d_hat).tocoo()  # a.dot(b) 指的是矩阵a与矩阵b的乘法
        # tocoo 就是  (0, 0)	0.25  (0, 633)	0.25 即稀疏矩阵三元组

    # ## 图卷积层定义


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta
        这个L就是 那个D-1/2AD-1/2 Symmetric normalized Laplacian 标准化的拉普拉斯矩阵
        这样就把傅里叶变化和拉普拉斯矩阵结合起来了.
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)  # https://blog.csdn.net/xinjieyuan/article/details/106951116
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
                Args:
                -------
                    adjacency: torch.sparse.FloatTensor
                        邻接矩阵
                    input_feature: torch.Tensor
                        输入特征
        """
        support = torch.mm(input_feature, self.weight)
        # torch.mul(a, b)
        # 是矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是维度(1, 2)
        # 的矩阵
        # torch.mm(a, b)
        # 是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的维度就是(1, 3)
        # 的矩阵
        output = torch.sparse.mm(adjacency, support)
        # sparse 因为 adjacency构造的时候是sparse的稀疏矩阵,所以使用稀疏矩阵乘法
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GcnNet(nn.Module):
    """
        定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim=1433):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16)
        self.gcn2 = GraphConvolution(16, 7)

    def forwaer(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits


# ## 模型训练

# In[5]:


# 超参数定义

learning_rate = 0.1
weight_decay = 5e-4  # 权重衰减https://blog.csdn.net/program_developer/article/details/80867468
epochs = 200

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GcnNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 加载数据，并转换为torch.Tensor
dataset = CoraData().data
x = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1,数据归一化后，最优解的寻优过程明显会变得平缓，更容易正确的收敛到最优解。,sum(1),1是按行,0是按列
tensor_x = torch.from_numpy(x).to(device)
tensor_y = torch.from_numpy(dataset.y).to(device)
tensor_train_mask = torch.from_numpy(dataset.train_mask).to(device)
tensor_val_mask = torch.from_numpy(dataset.val_mask).to(device)
tensor_test_mask = torch.from_numpy(dataset.test_mask).to(device)
normalize_adjacency = CoraData.normalization(dataset.adjacency)  # 规范化邻接矩阵
indices = torch.from_numpy(np.asarray([normalize_adjacency.row, normalize_adjacency.col]).astype('int64')).long
values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values, (2708, 2708)).to(device)


# 根据三元组 构造 稀疏矩阵向量,构造出来的张量是 (2708,2708)


# 训练主体函数
def train():
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(epochs):
        logits = model(tensor_adjacency, tensor_x)  # 前向传播
        train_mask_logits = logits[tensor_train_mask]  # 只选择训练节点进行监督
        loss = criterion(train_mask_logits,
                         train_y)  # 计算损失值 cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='elementwise_mean')
        optimizer.zero_grad()
        loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新,更新学习率
        train_acc, _, _ = tt(tensor_train_mask)  # 计算当前模型训练集上的准确率
        val_acc, _, _ = tt(tensor_val_mask)
        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))


def tt(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]  # torch.max返回两个向量,第一个向量是value,第二个向量是下表,所以有max(1)[1] 返回第二个向量的值,也就是下标
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float.mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history, c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history, c=np.array([79, 179, 255]) / 255.)
    plt.ylabel('ValAcc')
    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


loss, val_acc = train()
print(tensor_test_mask)
test_acc, test_logits, test_label = tt(tensor_test_mask)
print("Test accuarcy: ", test_acc.item())

