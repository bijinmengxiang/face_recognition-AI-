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

batch_size_train=10
batch_size_test=10
learning_rate = 0.015#学习率
momentum = 0.5#优化器参数

train_transform=transforms.Compose([
            transforms.CenterCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

test_transform=transforms.Compose([
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

train_dataset =torchvision.datasets.ImageFolder(root='G:\\train_sources\\face_database\\images\\images\\face\\train',transform=train_transform)
train_loader =DataLoader(train_dataset,batch_size=batch_size_train, shuffle=True,num_workers=0)

test_dataset =torchvision.datasets.ImageFolder(root='G:\\train_sources\\face_database\\images\\images\\face\\test',transform=train_transform)
test_loader =DataLoader(test_dataset,batch_size=batch_size_test, shuffle=True,num_workers=0)

class Net(nn.Module):
    def __init__(self):
        #nn.Module中的初始化函数
        super(Net, self).__init__()
        #84*84*3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        #80*80*6
        #self.avg_pool2 = F.avg_pool2d()
        #40*40*6
        self.sigmoid = nn.Sigmoid()
        #40*40*6
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        #36*36*16
        #self.avg_pool2 = F.avg_pool2d()
        #18*18*16
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=12, kernel_size=18)
        #1*1*12
        #x = x.view(-1, 12)
        #12
        self.linear1 = nn.Linear(12,4)
        #4
        
        #self.linear2 = nn.Linear(20,4)
        #1*1*4
        
    def forward(self,x):
        x=self.conv1(x)
        x = F.avg_pool2d(x,2)
        x=self.sigmoid(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x,2)
        x = self.conv3(x)
        #print(x.shape)
        x = x.view(-1,12)
        x = F.dropout(x,training=self.training)
        x = self.linear1(x)
        x = F.log_softmax(x)
        return x
    
def train():
    
    model.train()
    for i in range(80):
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
            #print(target)
            loss = F.nll_loss(predict, target)
            #进行反向传播
            loss.backward()
            #梯度更新
            optimizer.step()
            #每进行50批次训练输出一次结果
            if batch_idx%30==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        i, batch_idx * len(data), len(train_loader.dataset),
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
    
model = Net()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)

train()
test()
