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
    face_detecter = cv.CascadeClassifier(r'C:/Users/cnhhdn/Desktop/haarcascade_frontalface_default.xml')
    # 多个尺度空间进行人脸检测   返回检测到的人脸区域坐标信息
    try:
        faces = face_detecter.detectMultiScale(image=image, scaleFactor=1.1, minNeighbors=5)
    except:
        return
    #print('检测人脸信息如下：\n', faces)

    #路径无意义，为了创建两个图像类型变量
    chop = cv.imread("C:/Users/cnhhdn/Desktop/img/1.jpg")
    grid=image
    target=90
    for x, y, w, h in faces:
        print(x, y, w, h)
        line_x = x+w
        line_y = y+h
        # 在原图像上绘制矩形标识
        #cv.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
        #将红框图片显示
        #cv.imshow('final', image)
        #try为了检验错误
        try:
            if w < target:
                x = x - ((target - w) // 2) - 1
                w = w + ((target - w)) + 1
            if h < target:
                y = y - ((target - h) // 2) - 1
                h = h + ((target - h)) + 1

            if x<0 or y<0:
                if x<0 :
                    w = target
                else:
                    w =line_x
                if y<0:
                    h = target
                else:
                    h = line_y
                x = 0
                y = 0

            #将图片文件进行切割
            print(x, y, w, h)
            chop=grid[y:y+h,x:x+ w]
            cv.imwrite("G:\\train_sources\\face_database\\images\\images\\face\\test\\3\\" + str(count) + ".jpg", chop)
        except:
            print("切割错误")
        try:
            print(count)
            print("success")
            #cv.imshow('chop', chop)
        except:
            print("切割图片显示错误")

path = "G:\\train_sources\\face_database\\images\\images\\face\\test_dataset\\pengyuyan\\"   #图像读取地址
filelist = os.listdir(path)
count=0
for i in filelist:
    #将保存的文件进行打开操作
    img = cv.imread(path+str(i))
    #print(path+str(i))
    #调用函数进行检测
    #输入的count为接下来的文件名
    face_detection(img,count)
    count += 1
    cv.waitKey(0)
    cv.destroyAllWindows()


