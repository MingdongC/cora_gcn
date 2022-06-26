import scipy.sparse as sp
import torch
import gcn_net
import matplotlib.pyplot as plt
import numpy as np
import CoraData_process


###################################################################################
#                             模型训练与测试                                      #
###################################################################################

def train():
    loss_history = []
    val_acc_history = []
    train_acc_history = []

    gcn_net.model.train()
    train_y = CoraData_process.tensor_y[CoraData_process.tensor_train_mask]
    for epoch in range(gcn_net.epoches):
        logits = gcn_net.model(CoraData_process.tensor_adjacency, CoraData_process.tensor_x)  # 前向传播
        train_mask_logits = logits[CoraData_process.tensor_train_mask]  # 只选择训练节点进行监督
        loss = gcn_net.criterion(train_mask_logits, train_y)  # 计算损失值
        gcn_net.optimizer.zero_grad()
        loss.backward()  # 反向传播计算参数的梯度
        gcn_net.optimizer.step()  # 使用优化方法进行梯度更新
        train_acc = test(CoraData_process.tensor_train_mask)  # 计算当前模型在训练集上的准确率
        val_acc = test(CoraData_process.tensor_val_mask)  # 计算当前模型在验证集上的准确率

        # 记录训练过程中的损失值和准确率的变化，用于画图
        loss_history.append(loss.item())
        train_acc_history.append(train_acc.item())
        val_acc_history.append(val_acc.item())

        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4f}, ValAcc {:.4f}".format(epoch, loss.item(), train_acc.item(),
                                                                                 val_acc.item()))

    return loss_history, train_acc_history, val_acc_history


def test(mask):
    gcn_net.model.eval()
    with torch.no_grad():
        logits = gcn_net.model(CoraData_process.tensor_adjacency, CoraData_process.tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuracy = torch.eq(predict_y, CoraData_process.tensor_y[mask]).float().mean()
    return accuracy


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


############# 开始训练 ###################

print('=' * 15)
print('开始训练')

loss, train_acc, val_acc = train()  # 每个epoch 模型在训练集上的loss 和验证集上的准确率
# 可视化展示结果
plot_loss_with_acc(loss, val_acc)

# 计算最后训练好的模型在测试集上准确率
test_acc = test(CoraData_process.tensor_test_mask)
print("Test accuarcy: ", test_acc.item())

