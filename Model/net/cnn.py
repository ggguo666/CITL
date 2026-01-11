from torch.utils.data import Dataset
from torch import nn, reshape
import torch
import numpy as np
import random
import torch.nn.functional as F



class CNN(nn.Module):  # 服务器原版
    def __init__(self, p1, p2, c1, c2, h1, e1, dim=116, activation=nn.Tanh(), channel=1, re_activation=False,
                 instance_norm2=False):
        super(CNN, self).__init__()

        self.re_activation = re_activation  # 回归模型是否经过激活函数
        self.instance_norm2 = instance_norm2  # 是否对第二层卷积层的输出进行Instance_norm
        self.p1 = p1
        self.p2 = p2
        # GCN特征提取器参数
        self.conv1 = nn.Conv2d(channel, c1, (1, dim), 1, 0)  #
        self.conv2 = nn.Conv2d(c1, c2, (dim, 1), 1, 0)  #

        self.activation = activation
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(c2, h1)
        self.linear2 = nn.Linear(h1, 2)

        self.softmax = nn.Softmax()
        self.dropout1 = nn.Dropout(p1)
        self.dropout2 = nn.Dropout(p2)

        self.linear3 = nn.Linear(c2, e1)

        self.norm1 = nn.InstanceNorm2d(c1, affine=False)
        self.norm2 = nn.InstanceNorm2d(1, affine=False)

    def forward(self, x):
        x = x.to(torch.float32)
        FNC = x
        # 卷积
        conv1 = self.conv1(FNC)
        # print("conv1",conv1.shape)
        conv1 = self.activation(conv1)
        conv1 = self.norm1(conv1)

        conv2 = self.conv2(conv1)
        # print("conv2", conv2.shape)
        conv2 = self.activation(conv2)
        if self.instance_norm2 == True:
            conv2 = conv2.transpose(1, 2)
            conv2 = self.norm2(conv2)
        conv2 = self.flatten(conv2)
        # print(conv2.shape)
        # 全连接层1
        hidden1 = self.linear1(conv2)
        hidden1 = self.activation(hidden1)
        hidden1_drop = self.dropout1(hidden1)
        # 全连接层2
        hidden2 = self.linear2(hidden1_drop)
        hidden2 = self.softmax(hidden2)
        hidden2_drop = self.dropout2(hidden2)
        # 回归模型
        regression1 = self.linear3(conv2)
        if self.re_activation == True:
            regression1 = self.activation(regression1)
        return hidden2_drop, regression1
