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
