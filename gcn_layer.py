import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plot

############################################################################################
#                                      图卷积层的定义                                        #
############################################################################################

class GraphConvolution_layer(nn.Module):

    def __init__(self, input_dim, output_dim, bias_use=True):
        """
            卷积层定义为： X' = L*X*W
        :param input_dim:   输入节点的维度
        :param outputd_dim:     输出节点的维度
        :param bia_use:     是否使用偏置
        """
        super(GraphConvolution_layer,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias_use = bias_use
        self.weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.bias_use :
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    ###################初始化weights,bias############################
    def reset_parameters(self):
        init.kaiming_uniform_(self.weights)
        if self.bias_use:
            init.zeros_(self.bias)

    ##################定义前向传播函数forward#########################
    def forward(self, adjacency, features_matrix):
        """
            feature矩阵是稀疏矩阵，故前向传播采用torch.sparse.mm稀疏矩阵乘法
        :param adjacency:           输入节点的邻接矩阵，类型是torch.tensor
        :param features_matrix:     输入节点的特折矩阵，类型是torch.tensor
        :return:                    返回前向传播但未激活的output
        """

        temple = torch.mm(features_matrix, self.weights)
        output = torch.sparse.mm(adjacency, temple)

        ######若启用偏置，则输出加上偏置
        if self.bias_use:
            output += self.bias

        return output

