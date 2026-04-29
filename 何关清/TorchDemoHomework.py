#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026-04-29 7:56
@File    : TorchDemoHomework.py
"""

import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"""
基于pytorch 框架编写模型训练
任务： 实现一个自行构造的找规律的机器学习

随机生成一个9维向量最大的那个值的索引为输出  示例 [0, 1, 2, 3, 4, 5, 6, 7, 8], : 9
简单分析：
1. 这是一个多分类问题 激活使用Softmax，损失使用交叉熵
2. 线性层/全连接层 shape  9, 9
使用 numpy.random(9) 构造数据
"""


class TorchDemoHomework(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchDemoHomework, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)  # 这里使用一个隐藏扩宽网络提高表达能力
        self.activation = nn.ReLU()  # 添加激活
        self.linear2 = nn.Linear(128, output_size)
        self.loss = F.cross_entropy

    def forward(self, x, y=None):
        """前向传播"""
        x = self.linear1(x)

        x = self.activation(x)
        pred = self.linear2(x)
        if y is not None:
            # 由于交叉熵自带激活softmax，这里不需要再额外增加激活
            return self.loss(pred, y)

        else:
            #  如果你要直到预测结果的概率 搞事使用 softmax 进行处理
            # return F.softmax(pred, dim=1)  # pred 0为是batch, 我们要对最后一个做softmax， dim值为 1， 或者-1
            return pred  # 这里我也可以直接输出， 需要概率的话使用softmax


def build_sample():
    """构造单样本  真实值即为 x 的最大的索引"""
    x = np.random.random(9)
    return x, np.argmax(x)


def build_dataset(total_sample_num):
    """构造数据集"""
    X, Y = [], []

    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


def evaluate(model):
    """模型验证"""
    x, y = build_dataset(100)
    model.eval()  # 这个是固定写法 表示开始推理
    correct, wrong = 0, 0
    with torch.no_grad():  # 是验证，这里不必要进行计算梯度， 更新梯度
        y_pred = model(x)  # 前向计算 拿到预测值
        for pred, y_true in zip(y_pred, y):
            if pred.argmax().item() == y_true.item():
                correct += 1
            else:
                wrong += 1

    print("Accuracy: {:.2f}%".format(100 * correct / (correct + wrong)))
    return correct / (correct + wrong)


def train():
    """训练"""
    # 参数
    epoch_num = 200  # 训练多少轮次
    batch_size = 32  # 每次训练多少个样本数， 批大小
    train_sample_num = 5000  # 训练集数量
    input_size = 9
    output_size = 9
    learning_rate = 0.001  # 学习率
    # 训练集
    train_dataset, train_labels = build_dataset(train_sample_num)
    # 实例化模型
    model = TorchDemoHomework(input_size, output_size)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []  # 记录loss, acc
    start_time = time.time()
    for epoch in range(epoch_num):
        model.train()  # 这个是必须得， 开始训练
        watch_loss = []  # 记录每轮训完之后的loss
        for batch in range(train_sample_num // batch_size):
            x = train_dataset[batch * batch_size:(batch + 1) * batch_size]
            y = train_labels[batch * batch_size:(batch + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新梯度
            watch_loss.append(loss.item())
        print("Epoch {} Loss {}".format(epoch + 1, np.mean(watch_loss)))

        acc = evaluate(model)
        log.append([acc, np.mean(watch_loss)])
    print("Training Time {:.2f}".format(time.time() - start_time))

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')

    # 绘制acc 与loss
    plt.plot(range(len(log)), [l[0] for l in log], label='acc')
    plt.plot(range(len(log)), [l[1] for l in log], label='loss')
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    """推理"""
    input_size, output_size = 9, 9
    model = TorchDemoHomework(input_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 推理
    with torch.no_grad():  # 不计算梯度
        y_pred = model(input_vec)
        for pred, vec in zip(y_pred, input_vec):
            print(
                f"预测值：{pred.argmax().item()}--预测概率：{F.softmax(pred, dim=0)[pred.argmax().item()].item()} --真实值：{vec.argmax().item()}")


if __name__ == '__main__':
    train()
    input_vec, _  = build_dataset(10)
    predict(r"model.pth", input_vec)