
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.hub import load_state_dict_from_url
import random
import numpy as np
import os, shutil
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import csv
from torchvision import transforms

batch_size_train=64
batch_size_test=1000
learning_rate = 0.01#学习率
momentum = 0.5#优化器参数

#准备数据集
#训练数据集的加载器，自动将数据分割成batch，顺序随机打乱
#torch.utils.data.DataLoader(data数据集,batch_size每批次数量,shuffle重复训练后数据集是否洗牌)
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)
#测试集
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)


#enumerate函数 为字典中的值，赋顺序键。
examples = enumerate(test_loader)
#batch_idx为分批次训练中训练次数  example_data为此批次训练数据  example_targets为此批次数据目标
batch_idx, (example_data, example_targets) = next(examples)

#继承nn.Module
class Net(nn.Module):
    def __init__(self):
        #nn.Module中的初始化函数
        super(Net, self).__init__()
        
        #定义网络中的模块
        #28*28*1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,padding=2)
        #28*28*6
        #self.avg_pool1 = F.avg_pool2d()
        #14*14*6
        self.sigmoid = nn.Sigmoid()
        #14*14*6
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        #10*10*16
        #self.avg_pool2 = F.avg_pool2d()
        #5*5*16
        #由于特征图与卷积核大小相同，可看作为第一个线性层
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=12, kernel_size=5)
        #1*1*12*64
        self.linear1 = nn.Linear(12,10)
        #50
        self.linear2 = nn.Linear(50,10)
        #10
    
    #前向传播
    def forward(self,x):
        #28*28*1*64
        x = self.conv1(x)
        #28*28*6*64
        x = F.avg_pool2d(x,2)
        #14*14*6*64
        x = self.sigmoid(x)
        #14*14*6*64
        x = self.conv2(x)
        #10*10*16*64
        x = F.avg_pool2d(x,2)
        #5*5*16*64
        x = self.conv3(x)
        #print(x.shape)
        #1*1*12*64
        x = x.view(-1, 12)
        #64*12
        x = self.linear1(x)
        #64*10
        return F.log_softmax(x)
    
def train():
    
    model.train()
    
    for batch_idx,(data,target) in enumerate(train_loader):
        if torch.cuda.is_available() :
            model.cuda()
            target = target.cuda()
            data = data.cuda()
        #梯度置0
        optimizer.zero_grad()
        #计算预测值
        predict = model(data)
        #计算loss值
        print(target)
        loss = F.nll_loss(predict, target)
        #进行反向传播
        loss.backward()
        #梯度更新
        optimizer.step()
        #每进行50批次训练输出一次结果
        if batch_idx%50==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    torch.save(model, './test.pkl')
    
def test():
    #eval()保持权值不变
    #否则的话，有输入数据，即使不训练，它也会改变权值。
    #这是model中含有batch normalization层所带来的的性质。
    model.eval()
    
    test_loss = 0#测试平均loss值
    correct = 0#正确数
    #不进行梯度跟踪
    with torch.no_grad():
        for data,target in test_loader:
            #将训练数据转进gpu中
            if torch.cuda.is_available() :
                target = target.cuda()
                data = data.cuda()
            #使用模型进行预测
            predict = model(data)
            #计算总loss值
            test_loss += F.nll_loss(predict, target, size_average=False).item()
            #取可能性最大的值
            pred = predict.data.max(1, keepdim=True)[1]
            #正确数递增
            correct += pred.eq(target.data.view_as(pred)).sum()
    #求平均loss值（总loss值/测试集总数）
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    
#网络    
model = Net()
#优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)
#训练
train()
#测试
test()
        




















