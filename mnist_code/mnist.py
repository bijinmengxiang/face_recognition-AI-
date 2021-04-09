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

n_epochs = 3#训练循环次数
batch_size_train = 64#训练尺寸
batch_size_test = 1000#测试尺寸
learning_rate = 0.01#学习率
momentum = 0.5#优化器参数
log_interval = 100
random_seed = 1#设置随机种子
torch.manual_seed(random_seed)
path_ori="G:/train_sources/test/after_operator/singal_channel"

#载入数据集
#gz训练集
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
#example_targets训练集标签
#example_data训练集数据
batch_idx, (example_data, example_targets) = next(examples)
#print(batch_idx)
#print(example_data.shape)

class Net(nn.Module):
    #网络基本构架
    def __init__(self):
        super(Net, self).__init__()
        #卷积核5*5*10
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        #卷积核3*3*20
        self.conv2a = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2b = nn.Conv2d(20, 20 ,kernel_size=3)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #dropout层,提高鲁棒性，防止过拟合
        self.conv2_drop = nn.Dropout2d()
        #线性层320-》50
        self.fc1 = nn.Linear(320, 50)
        #self.fc1 = nn.Linear(2809, 20)
        #线性层50-》10
        self.fc2 = nn.Linear(50, 10)
        #self.fc2 = nn.Linear(20, 4)
    #前向传播过程    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) #x(1,10,110,110)
        x = self.conv2a(x)#x(1,20,108,108)
        x = self.conv2b(x)#x(1,20,106,106)
        x = F.relu(F.max_pool2d(self.conv2_drop(x),2))#x(64,20,4,4)
        #print(x.shape)
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print(x)
        x = x.view(-1, 320)
        print(x)
        #print(xyz)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



class Net1(nn.Module):
    #网络基本构架
    def __init__(self):
        super(Net1, self).__init__()
        #卷积核5*5*10
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5,stride=2)
        #卷积核5*5*20
        self.conv2a = nn.Conv2d(10, 20, kernel_size=3,stride=2)
        self.conv2b = nn.Conv2d(20, 20 ,kernel_size=3,stride=2)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #dropout层,提高鲁棒性，防止过拟合
        self.conv2_drop = nn.Dropout2d()
        #线性层56180-》2809
        #self.fc1 = nn.Linear(320, 50)
        self.fc1 = nn.Linear(720, 50)
        #线性层50-》10
        #self.fc2 = nn.Linear(50, 10)
        self.fc2 = nn.Linear(50, 4)
    #前向传播过程    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) #x(1,10,110,110)
        x = self.conv2a(x)#x(1,20,108,108)
        x = self.conv2b(x)#x(1,20,106,106)
        x = F.relu(F.max_pool2d(self.conv2_drop(x),2))#x(1,20,53,53)
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 720)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
#训练    
def train_fir_version(epoch):
    network.train()
    #从数据集中提取出需要的数据
    #遍历数据集
    #batch_idx 训练的次数
    for batch_idx, (data, target) in enumerate(train_loader):
        #梯度置0
        optimizer.zero_grad()
        #输出预测值
        output = network(data)
        #计算出loss值
        # output(64*10)
        # data(64,1,28,28)
        loss = F.nll_loss(output, target)
        #print(output.shape)
        #print(data.shape)
        #反向传播，修改参数
        loss.backward()
        #梯度更新
        optimizer.step()
    #训练输出数据监测
        #    print(batch_idx)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')

one=torch.ones(1)
loader = transforms.Compose([transforms.ToTensor()])
def train_sec_version():
    count=0
    network.train()
    #从数据集中提取出需要的数据
    #遍历数据集
    #batch_idx 训练的次数
    with open('G:/train_sources/test/data/images_ori.csv', 'r') as f:
     reader = csv.reader(f) 
     print(type(reader))
     loss_line=[]
     x_data=[]
     for row in reader:
         
         value=one*int(row[1])
         value=value.long()
         img=Image.open(path_ori+"/"+row[0])
         
         img = loader(img).unsqueeze(0)#unsqueeze增加维度
         optimizer.zero_grad()
         #img(1,1,224,224)
         #output(20,4)
         if use_cuda:
             img = img.cuda()
             value=value.cuda()
             network.cuda()
         output = network(img)
         #print(img.shape)
         loss = F.nll_loss(output, value)
         loss.backward()
         optimizer.step()
         count+=1
    
    #训练输出数据监测
        #    print(batch_idx)
         if count % log_interval == 0:
             print('Train process: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                count/log_interval, count, 14700,
                count/14700., loss.item()))
             train_losses.append(loss.item())
             #train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
             torch.save(network.state_dict(), './model.pth')
             torch.save(optimizer.state_dict(), './optimizer.pth')
#      
def test_sec():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
      with open('G:/train_sources/test/data/images.csv', 'r') as f:
          reader = csv.reader(f)
          for row in reader:
              
              value=one*int(row[1])
              value=value.long()
              #print(value)
              img=Image.open(path_ori+"/"+row[0])
              img = loader(img).unsqueeze(0)#unsqueeze增加维度
              if use_cuda:
                  img = img.cuda()
                  value=value.cuda()
                  network.cuda()
              #print(img.shape)
              output = network(img)
              #print(output.shape)
              test_loss += F.nll_loss(output, value, size_average=False).item()
              pred = output.data.max(1, keepdim=True)[1]
              correct += pred.eq(value.data.view_as(pred)).sum()
  test_loss /= 4700
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, 4700,
    100. * correct / 4700))
  
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
      
      for data, target in test_loader:
          #print(data.shape)
          output = network(data)
          #print(output.shape)
          test_loss += F.nll_loss(output, target, size_average=False).item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

#检测gpu是否可用
use_cuda = torch.cuda.is_available()
#定义网络    
network = Net()
#定义优化器
optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)

train_losses = []#训练集损失率
train_counter = []
test_losses = []#测试集损失率
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

 
#train_sec_version()
#test()
#test()       

for epoch in range(1, n_epochs + 1):
  train_fir_version(epoch)
        
import matplotlib.pyplot as plt
fig = plt.figure()
#plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        