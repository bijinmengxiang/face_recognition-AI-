# -*- coding: utf-8 -*-
"""
Created on Wed May 19 00:13:22 2021

@author: cnhhdn

This file descibe about how to build a spatial pyramid pooling 
"""
#正常定义math包用来上下取整
import math 
#size_stride函数作用：计算所需kernel_size以及stride（独立函数）
def size_stride(origin_size,target_size):  
    #向上取整
    size=math.ceil(origin_size/target_size)
    #向下取整
    stride=math.floor(origin_size/target_size)
    return size,stride

#以下代码写入forward中，也可在Net中进行封装。
#将stretch_final任意赋值
stretch_final=x

#使用循环[1,4) 后期可以使用数组代替实验
#i表示经过max_pool处理过后的尺寸
for i in range(1,4):
    #使用下列函数，计算出所需kernel_size以及stride
    size,stride=size_stride(x.shape[2],i)
    #使用对应kernel_size以及stride进行处理
    y=F.max_pool2d(x,kernel_size=size,stride=stride)
    #将处理过后的特征进行展开
    stretch = y.view(-1,32*i*i) 

    #首次将伸展值stretch赋给stretch_final
    if i==1:
        stretch_final=stretch
    #非首次，将伸展值stretch拼接至stretch_final上
    else:
        stretch_final = torch.cat((stretch_final,stretch),1)
#将stretch_final重新赋值给x不影响其他模块运作
x = stretch_final
