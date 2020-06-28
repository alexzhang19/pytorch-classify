#!/usr/bin/env python
# coding: utf-8

"""
@File      : lenet.py
@author    : alex
@Date      : 2020/6/28
@Desc      : https://blog.csdn.net/weixin_34910922/article/details/105694315
"""

import torch as t
import torch.nn as nn

__all__ = ["LeNet"]


class LeNet(nn.Module):
    def __init__(self, n_cls=10):
        super(LeNet, self).__init__()

        self.conv1 = nn.Sequential(  # input_size=(1*32*32)
            nn.Conv2d(1, 6, 5, 1, padding=0),  # padding=0,与原文保持一致
            nn.ReLU(inplace=True),  # input_size=(6*28*28)
            nn.MaxPool2d(2, 2),  # output_size=(6*14*14)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(84, n_cls)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)  # 按照batch大小展开
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


if __name__ == '__main__':
    x = t.randn(1, 1, 32, 32)  # 0~1正态分布
    model = LeNet()
    y = model(x)
    print(y, y.shape)
    pass
