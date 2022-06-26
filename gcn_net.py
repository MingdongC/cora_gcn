import torch.nn.functional as f
import torch.nn as nn
import torch
import torch.optim as optim
import gcn_layer

########################################################################################
#                                   模型构建                                            #
########################################################################################

class Gcn_net(nn.Module):

    """
        定义一个包含两层gcn_layer的模型
    """

    def __init__(self, input_dim=1433):

        super(Gcn_net,self).__init__()
        self.gcn1 = gcn_layer.GraphConvolution_layer(input_dim,18)
        self.gcn2 = gcn_layer.GraphConvolution_layer(18,7)

    def forwand(self, adjacency, feature_matrix):

        h = f.relu(self.gcn1(adjacency,feature_matrix))
        logits = self.gcn2(adjacency,h)

        return logits

learning_rate = 0.05
epoches = 500
weight_decay = 5e-4

# 参数设置
device = "cuda" if torch.cuda.is_available() else "cpu"


model = Gcn_net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
