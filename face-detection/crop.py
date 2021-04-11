import requests
import re
import urllib.parse
import cv2 as cv
import random
import glob
import matplotlib.pyplot as plt
import os

def face_detection(image,count):
    # 创建一个级联分类器 加载一个.xml分类器文件 它既可以是Haar特征也可以是LBP特征的分类器
    #此处xml文件路径需修改
    face_detecter = cv.CascadeClassifier(r'C:/Users/cnhhdn/Desktop/haarcascade_frontalface_default.xml')
    # 多个尺度空间进行人脸检测   返回检测到的人脸区域坐标信息
    faces = face_detecter.detectMultiScale(image=image, scaleFactor=1.1, minNeighbors=5)
    #print('检测人脸信息如下：\n', faces)

    #路径无意义，为了创建两个图像类型变量
    chop = cv.imread("C:/Users/cnhhdn/Desktop/img/1.jpg")
    grid=image
    for x, y, w, h in faces:
        # 在原图像上绘制矩形标识
        #cv.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
        #将红框图片显示
        #cv.imshow('final', image)
        try:
            #将图片文件进行切割
            chop=grid[y:y+h,x:x+ w]
            #图片保存路径
            cv.imwrite("G:\\train_sources\\face_database\\images\\images\\face\\jiangwen_change\\" + str(count) + ".jpg", chop)
        except:
            print("切割错误")
        try:
            print("success")
            #cv.imshow('chop', chop)
        except:
            print("切割图片显示错误")
            
#图像读取路径
path = "G:\\train_sources\\face_database\\images\\images\\face\\jiangwen\\"   
filelist = os.listdir(path)
count=0
for i in filelist:
    #将保存的文件进行打开操作
    src = cv.imread(path+str(i))
    #print(path+str(i))
    #调用函数进行检测
    #输入的count为接下来的文件名
    face_detection(src,count)
    count+=1
    cv.waitKey(0)
    cv.destroyAllWindows()
